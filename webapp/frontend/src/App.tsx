import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AnalysisProvider } from "./context/AnalysisContext";
import Landing from "./pages/Landing";
import Upload from "./pages/Upload";
import Workspace from "./pages/Workspace";

export default function App() {
  return (
    <AnalysisProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/workspace/:relayType/:analysisId" element={<Workspace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AnalysisProvider>
  );
}
