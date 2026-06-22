import fs from "fs";
import path from "path";
import os from "os";
import { glob } from "glob";
import { v4 as uuidv4 } from "uuid";

import { fromPath } from "pdf2pic";
import Tesseract from "tesseract.js";

import { RecursiveCharacterTextSplitter }
from "@langchain/textsplitters";

import { OllamaEmbeddings }
from "@langchain/ollama";

import { ChromaClient } from "chromadb";

const PDF_DIR = "./docs";
const COLLECTION_NAME = "company_docs";

const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text",
  baseUrl: process.env.OLLAMA_URL || "http://localhost:11434",
});

const chroma = new ChromaClient({
  host: process.env.CHROMA_HOST || "localhost",
  port: process.env.CHROMA_PORT || 8000,
});

async function getCollection() {
   return await chroma.getOrCreateCollection({
    name: COLLECTION_NAME,
    metadata: {
      "hnsw:space": "cosine",
    },
  });
}

async function ocrPdf(pdfPath) {
  const tempDir = path.join(
    os.tmpdir(),
    `ocr-${Date.now()}`
  );

  fs.mkdirSync(tempDir, { recursive: true });

  const converter = fromPath(pdfPath, {
    density: 300,
    saveFilename: "page",
    savePath: tempDir,
    format: "png",
    width: 2000,
    height: 2800,
  });

  let pages = [];
  let page = 1;

  while (true) {
    try {
      const result = await converter(page);

      if (!result?.path) break;

      pages.push({
        page,
        image: result.path,
      });

      page++;
    } catch {
      break;
    }
  }

  console.log(
    `${path.basename(pdfPath)} -> ${pages.length} pages`
  );

  const output = [];

  for (const p of pages) {
    console.log(
      `OCR page ${p.page}/${pages.length}`
    );

    const result = await Tesseract.recognize(
      p.image,
      "eng",
      {
        logger: m => {
          if (m.status === "recognizing text") {
            process.stdout.write(".");
          }
        }
      }
    );

    output.push({
      page: p.page,
      text: result.data.text,
    });
  }

  return output;
}

async function ingestPdf(pdfPath, collection) {

  const pages = await ocrPdf(pdfPath);

  const splitter =
    new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

  for (const page of pages) {

    const cleaned = page.text
      .replace(/\s+/g, " ")
      .trim();

    if (cleaned.length < 50) {
      continue;
    }

    const chunks =
      await splitter.splitText(cleaned);

    const vectors =
      await embeddings.embedDocuments(chunks);

    const ids = [];
    const docs = [];
    const metas = [];

    for (let i = 0; i < chunks.length; i++) {

      ids.push(uuidv4());

      docs.push(chunks[i]);

      metas.push({
        source: path.basename(pdfPath),
        page: page.page,
        chunk: i,
      });
    }

    await collection.add({
      ids,
      embeddings: vectors,
      documents: docs,
      metadatas: metas,
    });

    console.log(
      `Stored page ${page.page} (${chunks.length} chunks)`
    );
  }
}

async function main() {

  const collection =
    await getCollection();

  const files =
    await glob(`${PDF_DIR}/**/*.pdf`);

  console.log(
    `Found ${files.length} PDF files`
  );

  for (const pdf of files) {

    console.log(
      `\nProcessing ${path.basename(pdf)}`
    );

    await ingestPdf(pdf, collection);
  }

  console.log("\nFinished");
}

main().catch(console.error);