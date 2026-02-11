import re
import json
import os
import zipfile
import pandas as pd
import streamlit as st 
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Iterator, Tuple

# Internal Imports
# -- Setup Env from Secrets for Streamlit Cloud --
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
if "GOOGLE_API_KEY_2" in st.secrets:
    os.environ["GOOGLE_API_KEY_2"] = st.secrets["GOOGLE_API_KEY_2"]
if "TAVILY_API_KEY" in st.secrets:
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

from app_backend import app

# --- Helpers ---

def safe_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "blog"

def create_zip(md_content: str, filename: str, image_path: Path) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(filename, md_content.encode("utf-8"))
        if image_path.exists():
            for img in image_path.glob("*"):
                z.write(img, arcname=img.name)
    return buffer.getvalue()

def stream_execution(graph, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Yields step updates from the graph execution."""
    try:
        for output in graph.stream(inputs, stream_mode="updates"):
            yield "update", output
        final = graph.invoke(inputs)
        yield "final", final
    except Exception as e:
        yield "error", str(e)

def update_state(current: Dict[str, Any], update: Any) -> Dict[str, Any]:
    if isinstance(update, dict):
        if len(update) == 1 and isinstance(list(update.values())[0], dict):
            current.update(list(update.values())[0])
        else:
            current.update(update)
    return current

def render_preview(md: str):
    """Renders markdown with local image support."""
    # Split by image syntax to handle local paths
    parts = re.split(r'(!\[.*?\]\(.*?\))', md)
    for part in parts:
        if not part: continue
        img_match = re.match(r'!\[(.*?)\]\((.*?)\)', part)
        if img_match:
            alt, src = img_match.groups()
            path = Path(src)
            if path.exists():
                st.image(str(path), caption=alt)
            else:
                st.image(src, caption=alt) # Fallback for URLs
        else:
            st.markdown(part)

def load_history() -> List[Path]:
    return sorted(
        [p for p in Path(".").glob("*.md") if p.name.lower() != "readme.md"],
        key=lambda p: p.stat().st_mtime, 
        reverse=True
    )

# --- UI Setup ---

st.set_page_config(page_title="Blog Agent", layout="wide", page_icon="✍️")
st.title("Blog Writing Agent")

# Sidebar Controls
with st.sidebar:
    st.header("Controls")
    topic = st.text_area("Topic", height=100)
    target_date = st.date_input("Target Date", value=date.today())
    
    if st.button("Start Generation", type="primary"):
        st.session_state["is_running"] = True
        st.rerun()
        
    st.divider()
    
    # History Loader
    history = load_history()
    if history:
        st.subheader("History")
        selected_file = st.selectbox("Previous Blogs", options=history, format_func=lambda p: p.stem)
        if st.button("Load"):
            content = selected_file.read_text(encoding="utf-8")
            st.session_state["result"] = {"final": content}

# Main Execution Logic
# Main Execution Logic
if st.session_state.get("is_running", False) and topic:
    inputs = {
        "topic": topic,
        "as_of": target_date.isoformat(),
        "mode": "", # auto-decided
    }
    
    status_container = st.status("Processing...", expanded=True)
    metrics = st.empty()
    
    try:
        # Inject secrets if not present in env
        pass

        # Run Graph
        inputs = {"topic": topic, "as_of": target_date.strftime("%Y-%m-%d")}
        final_state = None
        
        for kind, payload in stream_execution(app, inputs):
            # Update UI based on event type
            if kind == "update":
                node = list(payload.keys())[0]
                status_container.write(f"Completed step: **{node}**")
            elif kind == "error":
                raise Exception(payload) # Rethrow to catch block
            elif kind == "final":
                final_state = payload

        # Success
        status_container.update(label="Complete", state="complete", expanded=False)
        st.session_state["is_running"] = False # Reset on success
        if final_state:
             st.session_state["result"] = final_state

    except Exception as e:
        status_container.update(label="Error", state="error", expanded=True)
        error_msg = str(e)
        
        # Handle Quota Exhaustion Gracefully
        if "exhausted" in error_msg.lower() or "quota" in error_msg.lower() or "429" in error_msg:
            st.error("⚠️ All API Keys have hit their limit for `gemini-2.5-flash`.")
            
            # Dynamic Key Injection
            with st.form("recovery_form"):
                new_key = st.text_input("Enter a temporary Google API Key to continue:", type="password")
                if st.form_submit_button("Retry with New Key"):
                     if new_key:
                        os.environ["GOOGLE_API_KEY"] = new_key
                        st.success("Key added! Retrying...")
                        st.rerun() # Rerun keeps is_running=True
        else:
            st.error(f"Execution failed: {e}")
            st.session_state["is_running"] = False

# Result Display
if "result" in st.session_state:
    result = st.session_state["result"]
    
    tab_view, tab_raw, tab_data = st.tabs(["Preview", "Raw Markdown", "Data"])
    
    with tab_view:
        render_preview(result.get("final", ""))
        
        # Downloads
        md_text = result.get("final", "")
        fname = safe_filename(topic[:20] if topic else "blog_post")
        
        col1, col2 = st.columns(2)
        col1.download_button("Download Markdown", md_text, f"{fname}.md")
        
        zip_bytes = create_zip(md_text, f"{fname}.md", Path("images"))
        col2.download_button("Download Bundle", zip_bytes, f"{fname}.zip", mime="application/zip")
        
    with tab_raw:
        st.code(result.get("final", ""), language="markdown")
        
    with tab_data:
        st.json(result)
