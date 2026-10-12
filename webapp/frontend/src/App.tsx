import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AnalysisProvider } from "./context/AnalysisContext";
import Landing from "./pages/Landing";

const DoubleEndedFL = lazy(() => import("./pages/DoubleEndedFL"));
const EventSimulator = lazy(() => import("./pages/EventSimulator"));
const IncidentList = lazy(() => import("./pages/IncidentList"));
const IncidentWorkspace = lazy(() => import("./pages/IncidentWorkspace"));
const TwsViewer = lazy(() => import("./pages/TwsViewer"));
const Upload = lazy(() => import("./pages/Upload"));
const Workspace = lazy(() => import("./pages/Workspace"));

export default function App() {
  return (
    <AnalysisProvider>
      <BrowserRouter>
        <Suspense fallback={<div role="status" style={{ padding: "2rem" }}>Loading page…</div>}>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/simulator" element={<EventSimulator />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/workspace/:relayType/:analysisId" element={<Workspace />} />
            <Route path="/tws/:analysisId" element={<TwsViewer />} />
            <Route path="/de-fl" element={<DoubleEndedFL />} />
            <Route path="/incidents" element={<IncidentList />} />
            <Route path="/incidents/:incidentId" element={<IncidentWorkspace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </AnalysisProvider>
  );
}
