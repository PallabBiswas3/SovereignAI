import React, { useState } from 'react';
import { extractTextFromPDF, validatePDFFile } from '../services/pdfService';
import { API_BASE_URL } from '../services/graphRagService';

interface Node {
  id: string;
  label: string;
  type: string;
  description?: string;
}

interface Link {
  source: string;
  target: string;
  type: string;
}

interface GraphData {
  nodes: Node[];
  links: Link[];
}

const PDFUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError('');
    const selectedFile = e.target.files?.[0] ?? null;
    if (selectedFile && validatePDFFile(selectedFile)) {
      setFile(selectedFile);
    } else {
      setError('Please select a valid PDF file (max 10MB).');
      setFile(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError('');
    try {
      // Extract text from PDF
      const text = await extractTextFromPDF(file);

      // Call backend API
      const res = await fetch(`${API_BASE_URL}/api/graph/extract`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.message || 'Failed to extract graph.');
      }

      const data: GraphData = await res.json();
      setGraph(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '1rem', maxWidth: '600px', margin: 'auto' }}>
      <h2>Upload PDF to Extract Knowledge Graph</h2>
      <input type="file" accept="application/pdf" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={!file || loading} style={{ marginLeft: '1rem' }}>
        {loading ? 'Processing...' : 'Extract Graph'}
      </button>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {graph && (
        <div style={{ marginTop: '1rem' }}>
          <h3>Extracted Graph JSON:</h3>
          <pre style={{ background: '#f0f0f0', padding: '1rem', overflowX: 'auto' }}>
            {JSON.stringify(graph, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};

export default PDFUpload;
