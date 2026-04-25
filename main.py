from rag_service import RAGService


def main() -> None:
    rag = RAGService()
    # rag.reset()
    # rag.add_documents(["docs/尺码推荐.txt", "docs/洗涤养护.txt", "docs/颜色选择.txt"])
    results = rag.search(
        "尺码",
        top_k=3,
        use_rerank=False,
        candidate_top_k=10,
    )

    print(results)


if __name__ == "__main__":
    main()