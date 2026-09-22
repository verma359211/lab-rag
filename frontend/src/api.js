// Vite exposes variables beginning with VITE_ to browser code. This URL is not
// a secret; it simply tells the browser where the FastAPI application lives.
const API_BASE_URL = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

function apiUrl(path) {
  if (!API_BASE_URL) {
    throw new Error("VITE_API_URL is missing from the frontend environment");
  }

  return `${API_BASE_URL}${path}`;
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(apiUrl("/ingest"), {
    method: "POST",
    body: formData,
  });

  return readResponse(response);
}

export async function askQuestion(question) {
  const response = await fetch(apiUrl("/chat"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  return readResponse(response);
}

export async function getDocuments() {
  const response = await fetch(apiUrl("/documents"));

  return readResponse(response);
}

export async function getDocumentChunks(source) {
  const query = new URLSearchParams({ source });
  const response = await fetch(apiUrl(`/documents/chunks?${query}`));

  return readResponse(response);
}

export async function deleteDocument(source) {
  const query = new URLSearchParams({ source });
  const response = await fetch(apiUrl(`/documents?${query}`), {
    method: "DELETE",
  });

  return readResponse(response);
}

async function readResponse(response) {
  const contentType = response.headers.get("content-type") || "";

  // A non-JSON response usually means the URL points to a web page or proxy
  // error instead of FastAPI. Give a useful message rather than JSON's vague
  // "Unexpected token '<'" parsing error.
  if (!contentType.includes("application/json")) {
    throw new Error(`API returned ${response.status} instead of JSON`);
  }

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || data.error || "Something went wrong");
  }

  return data;
}
