import { ChromaClient } from "chromadb";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";

const client = new ChromaClient({
  host: process.env.CHROMA_HOST || "localhost",
  port: process.env.CHROMA_PORT || 8000,
});

const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text",
});

const llm = new ChatOllama({
  model: "llama3.1",
  temperature: 0,
});

async function ask(question) {
  const collection = await client.getCollection({
    name: "company_docs",
  });

  const queryEmbedding =
    await embeddings.embedQuery(question);

  const results = await collection.query({
    queryEmbeddings: [queryEmbedding],
    nResults: 5,
  });

  const context =
    results.documents[0].join("\n\n");

  const response = await llm.invoke(`
You are a company knowledge assistant.

Answer only from the provided context.

Context:
${context}

Question:
${question}
`);

  console.log(response.content);

  console.log(
    "\nSources:",
    results.metadatas[0]
  );
}

await ask(
    "When is the Eid Milad Un Nabi holiday?"

    // "What date will the company be closed?"

    // "Who issued the announcement?"

    // "What is the subject of the circular?"

    // "Which departments is the notice addressed to?"

    // "What are the instructions for employees working on the holiday?"

    // "Will employees receive overtime for working on the holiday?"

    // "Who signed the document?"

    // "What is the designation of Salman Kafeel?"

    // "What company issued this notice?"

    // "What is the company's office address?"

    // "What is the company's website?"

    // "What is the company's email address?"

    // "What is the office telephone number?"

    // "Where is the corporate office located?"

    // "Who is the Chief Operating Officer?"

    // "Who is the Chief Marketing Officer?"

    // "Summarize this document."

    // "Give me a short summary of the holiday announcement."

    // "What are the key points in this notice?"

    // "Explain this circular in simple language."

    // "What actions are required from employees?"

    // "What is the holiday policy mentioned in this document?"

    // "How will employees be compensated if they work on the holiday?"

    // "Are all outlets closed during the holiday?"

    // "Who determines the duty schedule for the holiday?"

    // "What guidelines should employees follow during the holiday period?"

    // "Which PDF contains information about Eid Milad Un Nabi?"

    // "From which page was this answer retrieved?"

    // "Show the source document."

    // "Which document mentions overtime compensation?"

    // "Which page discusses holiday schedules?"

    // "When do employees get a day off for a religious occasion?"

    // "What happens if staff work during the public holiday?"

    // "Who should employees contact regarding holiday schedules?"

    // "Is there any compensation for holiday duty?"

    // "Tell me about the company's holiday arrangements."

    // "Are stores open during the holiday?"

    // "What instructions were given to outlet employees?"

    // "What announcement was made for September 2025?"

    // "Compare leave policies across all documents."

    // "Which documents mention public holidays?"

    // "List all announcements issued in 2025."

    // "Find documents related to employee benefits."

    // "What company policies are mentioned across the knowledge base?"

    // "Show all references to overtime."

    // "Which documents were issued by management?"

    // "Summarize all HR-related documents."

    // "When is the company closed for Eid Milad Un Nabi?"

    // "What compensation is provided to employees who work on the holiday?"

    // "Who issued the circular and what position do they hold?"

    // "Summarize the holiday announcement in 3 bullet points."
);