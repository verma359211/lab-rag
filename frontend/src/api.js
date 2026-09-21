export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/ingest", {
    method: "POST",
    body: formData,
  });

  return readResponse(response);
}

export async function askQuestion(question) {
  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  return readResponse(response);
}

async function readResponse(response) {
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || data.error || "Something went wrong");
  }

  return data;
}
