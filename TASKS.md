# LectureMindAI Build Checklist

- [x] 1. Project scaffolding
- [x] 2. Backend skeleton
- [x] 3. PDF upload + text extraction
- [x] 4. Chunking utility
- [x] 5. Embedding + ChromaDB text_chunks
- [x] 6. Ollama integration
- [x] 7. /query endpoint (PDF RAG)
- [x] 8. Audio ingestion
- [x] 9. Video ingestion
- [x] 10. URL ingestion (yt-dlp)
- [x] 11. Cross-modal retrieval
- [x] 12. SQLite chat history
- [x] 13. /documents list + DELETE
- [x] 14. Frontend scaffold
- [x] 15. Upload UI
- [x] 16. Chat UI
- [x] 17. Document sidebar
- [x] 18. Backend tests
- [x] 19. Security hardening
- [x] 20. Documentation
- [x] 21. Docker
- [x] 22. Deployment docs

## Evaluation and model tuning
- [x] 23. Evaluation harness (retrieval, thresholds, generation, hallucination, conversation)
- [x] 24. Real-data runs on QASPER, SQuAD v2, CoQA, RAGTruth, RAGBench
- [x] 25. Fine-tune NLI grounding checker on RAGBench train (RAGTruth AUROC 0.65 -> 0.74; no gain on RAGBench test, so not the default)
- [x] 26. Hard-negative embedder fine-tuning script
- [ ] 27. Run reranker and embedder fine-tunes, re-evaluate retrieval and re-calibrate thresholds
- [ ] 28. Improve the grounding checker (more data/epochs, MiniCheck or HHEM), re-tune GROUNDEDNESS_THRESHOLD
- [ ] 29. Hand-label 30-50 questions on real lecture PDFs (in-domain metric)
- [ ] 30. Larger-n runs (n>=500) with confidence intervals; compare qwen2.5:7b / llama3.1:8b
