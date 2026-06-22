import { ChromaClient } from "chromadb";

const client = new ChromaClient({
  host: process.env.CHROMA_HOST || "localhost",
  port: process.env.CHROMA_PORT || 8000,
});

try {
  await client.deleteCollection({
    name: "company_docs",
  });

  console.log("Deleted company_docs");
} catch (e) {
  console.log("Collection does not exist");
}