import faiss

index = faiss.IndexFlatL2(128)

print(index.is_trained)
print(index.ntotal)