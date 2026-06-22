import { StateGraph, START, END } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";
import { ChromaClient } from "chromadb";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";

/* -----------------------------
   Chroma + Ollama Setup
------------------------------*/

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

/* -----------------------------
   LangGraph State
------------------------------*/

const State = Annotation.Root({
  question: Annotation(),
  rewrittenQuestion: Annotation(),
  docs: Annotation(),
  answer: Annotation(),
  sources: Annotation(),
});

/* -----------------------------
   1. Query Rewriting Node
------------------------------*/

async function rewriteNode(state) {
  const prompt = `
Rewrite the question to improve search in a vector database.
Keep it short and semantic.

Question: ${state.question}
`;

  const res = await llm.invoke(prompt);

  return {
    rewrittenQuestion: res.content.trim(),
  };
}

/* -----------------------------
   2. Retrieval Node (Chroma)
------------------------------*/

async function retrieveNode(state) {
  const query = state.rewrittenQuestion || state.question;

  const collection = await client.getCollection({
    name: "company_docs",
  });

  const queryEmbedding = await embeddings.embedQuery(query);

  const results = await collection.query({
    queryEmbeddings: [queryEmbedding],
    nResults: 5,
  });

  return {
    docs: results.documents[0],
    sources: results.metadatas[0],
  };
}

/* -----------------------------
   3. Rerank / Filter Node
   (simple but effective)
------------------------------*/

async function rerankNode(state) {
  // remove very low quality chunks
  const filteredDocs = state.docs.filter(
    (d) => d && d.length > 20
  );

  return {
    docs: filteredDocs,
  };
}

/* -----------------------------
   4. Generation Node
------------------------------*/

async function generateNode(state) {
  const context = state.docs.join("\n\n");

  const prompt = `
You are a strict company assistant.

Rules:
- Answer ONLY from context
- If not found, say "Not found in documents"
- Always be precise

Context:
${context}

Question:
${state.question}
`;

  const res = await llm.invoke(prompt);

  return {
    answer: res.content,
  };
}

/* -----------------------------
   5. Validation Node
   (simple grounding check)
------------------------------*/

async function validateNode(state) {
  const hasContext = state.docs?.length > 0;

  if (!hasContext) {
    return {
      answer: "Not found in documents",
    };
  }

  return {};
}

/* -----------------------------
   Build Graph
------------------------------*/

const graph = new StateGraph(State)
  .addNode("rewrite", rewriteNode)
  .addNode("retrieve", retrieveNode)
  .addNode("rerank", rerankNode)
  .addNode("generate", generateNode)
  .addNode("validate", validateNode)

/* Flow */
  .addEdge(START, "rewrite")
  .addEdge("rewrite", "retrieve")
  .addEdge("retrieve", "rerank")
  .addEdge("rerank", "generate")
  .addEdge("generate", "validate")
  .addEdge("validate", END)

  .compile();

/* -----------------------------
   Run Example
------------------------------*/

const result = await graph.invoke({
    // question: "When is Eid Milad Un Nabi holiday?",

     // "What date will the company be closed?"

    // "Who issued the announcement?"

    // "What is the subject of the circular?"

    // "Which departments is the notice addressed to?"

    // "What are the instructions for employees working on the holiday?"

    // "Will employees receive overtime for working on the holiday?"

    question: "Who signed the document?"

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

    // question: "Summarize the holiday announcement in 3 bullet points."
  
});

console.log("\nANSWER:\n", result.answer);

console.log("\nSOURCES:\n", result.sources);