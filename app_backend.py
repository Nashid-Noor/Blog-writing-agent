from __future__ import annotations

import os
import re
import operator
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# --- Configuration ---
PRIMARY_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.0-flash-exp"

# --- Data Models ---

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="Task objective.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)

class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)

class ImageSpec(BaseModel):
    placeholder: str
    filename: str
    alt: str
    caption: str
    prompt: str
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"

class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str

# --- Core Logic & Fallback & Key Rotation ---

def get_llm(model_name: str, api_key: str = None):
    """Instantiates the LLM with the given model name and specific API key."""
    # If no key provided, let it try default env var or fail
    if api_key:
        return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)
    return ChatGoogleGenerativeAI(model=model_name, temperature=0)

def get_keys():
    """Returns a list of available API keys."""
    keys = []
    if os.getenv("GOOGLE_API_KEY"):
        keys.append(os.getenv("GOOGLE_API_KEY"))
    if os.getenv("GOOGLE_API_KEY_2"):
        keys.append(os.getenv("GOOGLE_API_KEY_2"))
    return keys if keys else [None] # Return at least one None to try default env

def robust_call(messages: list, output_structure=None):
    """
    Calls the LLM with the PRIMARY_MODEL using available keys.
    If a key fails with quota error, rotates to the next key.
    If ALL keys fail for PRIMARY_MODEL, falls back to FALLBACK_MODEL (standard env key).
    """
    available_keys = get_keys()
    
    # -- Strategy 1: Primary Model + Key Rotation --
    for i, key in enumerate(available_keys):
        try:
            model = get_llm(PRIMARY_MODEL, api_key=key)
            if output_structure:
                chain = model.with_structured_output(output_structure)
                return chain.invoke(messages)
            else:
                return model.invoke(messages)
        except Exception as e:
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                print(f"⚠️ Quota exceeded for {PRIMARY_MODEL} on Key #{i+1}. Rotating...")
                continue # Try next key
            else:
                raise e # Fatal error (not quota)

    # -- Both Keys Exhausted --
    raise RuntimeError(f"All API keys exhausted for {PRIMARY_MODEL}. Please wait or add more keys.")

def router_node(state: State) -> dict:
    """Decides if research is needed and sets the operational mode."""
    system_prompt = (
        "You are a blog planner. Decide research needs based on the topic.\n"
        "Modes: closed_book (evergreen), hybrid (needs examples), open_book (news/recent events).\n"
        "If researching, provide 3-10 targeted queries."
    )
    
    decision = robust_call(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Topic: {state['topic']}\nDate: {state['as_of']}")
        ],
        output_structure=RouterDecision
    )

    recency_map = {"open_book": 7, "hybrid": 45, "closed_book": 3650}
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_map.get(decision.mode, 3650),
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
            for r in results or []
        ]
    except Exception:
        return []

def research_node(state: State) -> dict:
    """Executes search queries and synthesizes evidence."""
    queries = (state.get("queries") or [])[:10]
    raw_results = []
    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=6))

    if not raw_results:
        return {"evidence": []}

    system_prompt = (
        "Synthesize search results into structured evidence items.\n"
        "Filter for relevance and authority. Deduplicate by URL."
    )
    
    pack = robust_call(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Date: {state['as_of']}\nResults:\n{raw_results}")
        ],
        output_structure=EvidencePack
    )

    # Dedup and filter by date if strictly open_book
    evidence_map = {e.url: e for e in pack.evidence if e.url}
    evidence = list(evidence_map.values())
    
    if state.get("mode") == "open_book":
        cutoff = date.fromisoformat(state["as_of"]) - timedelta(days=state["recency_days"])
        evidence = [
            e for e in evidence 
            if e.published_at and (d := date.fromisoformat(e.published_at[:10]) if len(e.published_at) >= 10 else None) and d >= cutoff
        ]
    
    return {"evidence": evidence}

def orchestrator_node(state: State) -> dict:
    """Generates the blog outline."""
    system_prompt = (
        "Create a detailed blog outline. 5-9 sections.\n"
        "For 'open_book' mode, focus on news/analysis relying on provided evidence."
    )
    
    evidence_dump = [e.model_dump() for e in state.get("evidence", [])][:16]
    
    plan = robust_call(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Topic: {state['topic']}\nMode: {state.get('mode')}\nEvidence:\n{evidence_dump}")
        ],
        output_structure=Plan
    )
    
    if state.get("mode") == "open_book":
        plan.blog_kind = "news_roundup"
        
    return {"plan": plan}

def fanout(state: State):
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "plan": state["plan"].model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
        }) for task in state["plan"].tasks
    ]

def worker_node(payload: dict) -> dict:
    """Writes a single blog section."""
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    
    system_prompt = (
        "Write a blog section in Markdown. stick to the bullets and constraints.\n"
        "Cite sources using Markdown links if required."
    )
    
    response = robust_call(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Blog: {plan.blog_title}\nSection: {task.title}\nBullets: {task.bullets}\nEvidence: {evidence}")
        ]
    )
    
    return {"sections": [(task.id, response.content.strip())]}

def merge_content(state: State) -> dict:
    ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    full_md = f"# {state['plan'].blog_title}\n\n" + "\n\n".join(ordered)
    return {"merged_md": full_md}

def decide_images(state: State) -> dict:
    """Proposes where to insert images and generates prompts for them."""
    system_prompt = (
        "Act as an editor. Insert image placeholders [[IMAGE_X]] where diagrams/visuals would help.\n"
        "Max 3 images. Return the modified markdown and image specifications."
    )
    
    image_plan = robust_call(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["merged_md"])
        ],
        output_structure=GlobalImagePlan
    )
    
    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }

def _generate_image(prompt: str) -> bytes:
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    
    # Handle response structure variations
    parts = getattr(resp, "parts", None) or (resp.candidates[0].content.parts if resp.candidates else None)
    if not parts:
        raise RuntimeError("No image returned")
        
    for part in parts:
        if part.inline_data and part.inline_data.data:
            return part.inline_data.data
            
    raise RuntimeError("No inline image data found")

def generate_images(state: State) -> dict:
    """Generates images and saves them locally."""
    md = state.get("md_with_placeholders") or state["merged_md"]
    specs = state.get("image_specs", [])
    
    if not specs:
        Path(f"{_safe_slug(state['plan'].blog_title)}.md").write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)
    
    for spec in specs:
        path = images_dir / spec["filename"]
        if not path.exists():
            try:
                data = _generate_image(spec["prompt"])
                path.write_bytes(data)
                img_md = f"![{spec['alt']}](images/{spec['filename']})\n*{spec['caption']}*"
                md = md.replace(spec["placeholder"], img_md)
            except Exception as e:
                md = md.replace(spec["placeholder"], f"> *Image generation failed: {e}*")
        else:
             # Already exists
             img_md = f"![{spec['alt']}](images/{spec['filename']})\n*{spec['caption']}*"
             md = md.replace(spec["placeholder"], img_md)

    Path(f"{_safe_slug(state['plan'].blog_title)}.md").write_text(md, encoding="utf-8")
    return {"final": md}

def _safe_slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "blog"

# --- Graph Construction ---

workflow = StateGraph(State)
workflow.add_node("router", router_node)
workflow.add_node("research", research_node)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("worker", worker_node)

# Subgraph for reduction & images
reducer = StateGraph(State)
reducer.add_node("merge", merge_content)
reducer.add_node("plan_images", decide_images)
reducer.add_node("gen_images", generate_images)
reducer.add_edge(START, "merge")
reducer.add_edge("merge", "plan_images")
reducer.add_edge("plan_images", "gen_images")
reducer.add_edge("gen_images", END)

workflow.add_node("reducer", reducer.compile())

workflow.add_edge(START, "router")
workflow.add_conditional_edges("router", route_next)
workflow.add_edge("research", "orchestrator")
workflow.add_conditional_edges("orchestrator", fanout, ["worker"])
workflow.add_edge("worker", "reducer")
workflow.add_edge("reducer", END)

app = workflow.compile()
