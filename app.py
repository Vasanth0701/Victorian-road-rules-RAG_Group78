import os
import sys

import streamlit as st

# Let Python find rag_generate.py and the retrieval package (inside src/)
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from rag_generate import ask_road_rules

st.title("Victorian Road Rules RAG Assistant")
st.caption("Ask a question about Victorian road rules. Not legal advice: confirm with VicRoads.")

question = st.text_input("Your question")

if st.button("Ask") and question:
    with st.spinner("Searching road rules..."):
        result = ask_road_rules(question)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    if result["sources"]:
        for src in result["sources"]:
            line = f"**{src['title']}** - {src['section']}"
            if src.get("url"):
                line += f"  \n[{src['url']}]({src['url']})"
            st.markdown(line)
    else:
        st.info("No sources found for this question.")