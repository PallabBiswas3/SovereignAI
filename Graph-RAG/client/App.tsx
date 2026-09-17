import React, { useEffect, useReducer, useCallback } from "react";
import Sidebar from "./layout/Sidebar";
import Header from "./layout/Header";
import GraphWorkspace from "./graph/GraphWorkspace";
import IngestionPanel from "./ingestion/IngestionPanel";
import ChatPanel from "./chat/ChatPanel";

import {
  extractKnowledgeGraph,
  fetchGraphData,
  clearGraphData,
} from "./services/graphRagService";

import {
  GraphData,
  Node,
  ChatMessage,
  View,
  LoadingState,
  IngestionProgress,
} from "./types";

interface AppState {
  view: View;
  graph: GraphData;
  selectedNode: Node | null;
  messages: ChatMessage[];
  loading: LoadingState;
  ingestionProgress: IngestionProgress | null;
}

type Action =
  | { type: "SET_VIEW"; payload: View }
  | { type: "SET_GRAPH"; payload: GraphData }
  | { type: "SET_SELECTED_NODE"; payload: Node | null }
  | { type: "SET_MESSAGES"; payload: React.SetStateAction<ChatMessage[]> }
  | { type: "CLEAR_MESSAGES" }
  | { type: "SET_LOADING"; payload: Partial<LoadingState> }
  | { type: "SET_INGESTION_PROGRESS"; payload: IngestionProgress | null };

const initialState: AppState = {
  view: "input",
  graph: { nodes: [], links: [] },
  selectedNode: null,
  messages: [],
  loading: { extract: false, rag: false, pdf: false },
  ingestionProgress: null,
};

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "SET_VIEW":
      return { ...state, view: action.payload };
    case "SET_GRAPH":
      return { ...state, graph: action.payload };
    case "SET_SELECTED_NODE":
      return { ...state, selectedNode: action.payload };
    case "SET_MESSAGES": {
      const nextMessages =
        typeof action.payload === "function"
          ? action.payload(state.messages)
          : action.payload;
      return { ...state, messages: nextMessages };
    }
    case "CLEAR_MESSAGES":
      return { ...state, messages: [] };
    case "SET_LOADING":
      return { ...state, loading: { ...state.loading, ...action.payload } };
    case "SET_INGESTION_PROGRESS":
      return { ...state, ingestionProgress: action.payload };
    default:
      return state;
  }
}

const App: React.FC = () => {
  const [state, dispatch] = useReducer(reducer, initialState);

  const refreshGraph = useCallback(async () => {
    const graph = await fetchGraphData();
    dispatch({ type: "SET_GRAPH", payload: graph });
  }, []);

  useEffect(() => {
    refreshGraph().catch(console.error);
  }, [refreshGraph]);

  const handleViewChange = useCallback((view: View) => {
    dispatch({ type: "SET_VIEW", payload: view });
  }, []);

  const handleNodeSelect = useCallback((node: Node | null) => {
    dispatch({ type: "SET_SELECTED_NODE", payload: node });
  }, []);

  const handleExtract = useCallback(
    async (text: string, sourceDocId?: string) => {
      if (!text.trim()) return;
      dispatch({ type: "SET_LOADING", payload: { extract: true } });
      dispatch({ type: "SET_VIEW", payload: "graph" });
      try {
        const graph = await extractKnowledgeGraph(text, sourceDocId);
        dispatch({ type: "SET_GRAPH", payload: graph });
      } finally {
        dispatch({ type: "SET_LOADING", payload: { extract: false } });
      }
    },
    []
  );

  const handleIngestionProgress = useCallback(
    (progress: IngestionProgress | null) => {
      dispatch({ type: "SET_INGESTION_PROGRESS", payload: progress });
      if (progress?.status === "done" || progress?.status === "error") {
        setTimeout(
          () => dispatch({ type: "SET_INGESTION_PROGRESS", payload: null }),
          3000
        );
      }
    },
    []
  );

  const handlePdfLoading = useCallback((loading: boolean) => {
    dispatch({ type: "SET_LOADING", payload: { pdf: loading } });
  }, []);

  const handleClearGraph = useCallback(async () => {
    await clearGraphData();
    await refreshGraph();
    dispatch({ type: "CLEAR_MESSAGES" });
    dispatch({ type: "SET_SELECTED_NODE", payload: null });
    dispatch({ type: "SET_INGESTION_PROGRESS", payload: null });
  }, [refreshGraph]);

  const setMessages = useCallback(
    (action: React.SetStateAction<ChatMessage[]>) => {
      dispatch({ type: "SET_MESSAGES", payload: action });
    },
    []
  ) as React.Dispatch<React.SetStateAction<ChatMessage[]>>;

  const setRagLoading = useCallback(
    (action: React.SetStateAction<boolean>) => {
      const next = typeof action === "function" ? action(state.loading.rag) : action;
      dispatch({ type: "SET_LOADING", payload: { rag: next } });
    },
    [state.loading.rag]
  ) as React.Dispatch<React.SetStateAction<boolean>>;

  return (
    <div
      className="flex h-screen antialiased overflow-hidden"
      style={{ background: "#050508", color: "#e2e8f0" }}
    >
      <Sidebar
        active={state.view}
        onChange={handleViewChange}
        onClear={handleClearGraph}
        ingestionProgress={state.ingestionProgress}
        pdfLoading={state.loading.pdf}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <Header
          graph={state.graph}
          loading={state.loading.extract}
          pdfLoading={state.loading.pdf}
          ingestionProgress={state.ingestionProgress}
        />

        <main className="flex-1 overflow-hidden relative">
          {state.view === "input" && (
            <IngestionPanel
              loading={state.loading.extract}
              pdfLoading={state.loading.pdf}
              onSubmit={handleExtract}
              onPdfLoading={handlePdfLoading}
              onIngestionProgress={handleIngestionProgress}
              onViewChange={handleViewChange}
              onGraphReady={refreshGraph}
            />
          )}

          {state.view === "graph" && (
            <GraphWorkspace
              graph={state.graph}
              selectedNode={state.selectedNode}
              onSelect={handleNodeSelect}
            />
          )}

          {state.view === "chat" && (
            <ChatPanel
              messages={state.messages}
              setMessages={setMessages}
              loading={state.loading.rag}
              setLoading={setRagLoading}
              graph={state.graph}
              onClose={() => handleViewChange("graph")}
            />
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
