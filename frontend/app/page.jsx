'use client';

import { useMemo, useState } from 'react';

const MODE_OPTIONS = [
  {
    id: 'fast',
    title: 'Fast',
    note: 'Standard model with horizontal-flip TTA for quicker inspection.',
  },
  {
    id: 'accurate',
    title: 'Accurate',
    note: 'Standard inference + SAHI slicing + WBF for fine cracks.',
  },
];

function formatResolution(payload) {
  if (!payload) return '—';
  return `${payload.image_width} × ${payload.image_height}`;
}

export default function Page() {
  const [selectedMode, setSelectedMode] = useState('accurate');
  const [confidence, setConfidence] = useState(0.5);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const detectionChips = useMemo(() => {
    if (!result?.detections) return [];
    return result.detections.map((item) => `${item.class_name} ${item.confidence.toFixed(2)}`);
  }, [result]);

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    setSelectedFile(file || null);
    setError('');
    setResult(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(file ? URL.createObjectURL(file) : '');
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!selectedFile) {
      setError('Upload a road image first.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('mode', selectedMode);
      formData.append('confidence', confidence.toFixed(2));

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || 'Prediction request failed.');
      }

      setResult(payload);
    } catch (requestError) {
      setError(requestError.message || 'Prediction failed.');
    } finally {
      setLoading(false);
    }
  }

  const annotatedPreview = result
    ? `data:image/jpeg;base64,${result.annotated_image_base64}`
    : previewUrl;

  return (
    <main className="shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">CrackVision / road inspection system</p>
          <h1>Road damage detection with a field-calibrated visual language.</h1>
          <p className="lead">
            Inspect pavement imagery with a model tuned for thin cracks, potholes, and surface corruption.
            The interface stays operational, not decorative.
          </p>
        </div>

        <div className="hero-panel">
          <div className="stat">
            <span>Model</span>
            <strong>YOLOv8-L</strong>
          </div>
          <div className="stat">
            <span>Confidence</span>
            <strong>{confidence.toFixed(2)}</strong>
          </div>
          <div className="stat">
            <span>Mode</span>
            <strong>{MODE_OPTIONS.find((item) => item.id === selectedMode)?.title}</strong>
          </div>
        </div>
      </section>

      <section className="content-grid">
        <form className="control-panel" onSubmit={handleSubmit}>
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Inspection bay</p>
              <h2>Upload a road image</h2>
            </div>
          </div>

          <label className="dropzone">
            <input type="file" accept="image/*" onChange={handleFileChange} />
            <span className="dropzone-title">Drag an image here or browse</span>
            <span className="dropzone-subtitle">
              Works best on street-level frames, dashcam shots, and large road panoramas.
            </span>
          </label>

          <div className="segmented-control">
            {MODE_OPTIONS.map((option) => (
              <button
                key={option.id}
                type="button"
                className={selectedMode === option.id ? 'segment active' : 'segment'}
                onClick={() => setSelectedMode(option.id)}
              >
                <strong>{option.title}</strong>
                <span>{option.note}</span>
              </button>
            ))}
          </div>

          <label className="slider-block">
            <div className="slider-topline">
              <span>Confidence threshold</span>
              <strong>{confidence.toFixed(2)}</strong>
            </div>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.01"
              value={confidence}
              onChange={(event) => setConfidence(parseFloat(event.target.value))}
            />
          </label>

          <button className="primary-button" type="submit" disabled={loading}>
            {loading ? 'Analyzing frame...' : 'Run CrackVision'}
          </button>

          {error ? <p className="error-box">{error}</p> : null}
        </form>

        <section className="preview-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Output stage</p>
              <h2>Annotated result</h2>
            </div>
            <div className="result-pill">{result ? `${result.num_detections} detections` : 'Waiting for upload'}</div>
          </div>

          <div className="preview-frame">
            {annotatedPreview ? (
              <img src={annotatedPreview} alt="Annotated road image" />
            ) : (
              <div className="empty-preview">
                <span>Preview will appear here.</span>
              </div>
            )}
          </div>

          <div className="metrics-row">
            <div className="metric-card">
              <span>Mode</span>
              <strong>{result?.mode || selectedMode}</strong>
            </div>
            <div className="metric-card">
              <span>Threshold</span>
              <strong>{result?.confidence_threshold?.toFixed?.(2) ?? confidence.toFixed(2)}</strong>
            </div>
            <div className="metric-card">
              <span>Resolution</span>
              <strong>{formatResolution(result)}</strong>
            </div>
          </div>

          <div className="summary-card">
            <h3>System summary</h3>
            <p>{result?.summary || 'Run detection to inspect damage classes, confidence, and box geometry.'}</p>
          </div>
        </section>

        <aside className="details-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Detection log</p>
              <h2>Box stream</h2>
            </div>
          </div>

          <div className="chip-cloud">
            {detectionChips.length ? detectionChips.map((chip) => <span key={chip}>{chip}</span>) : <span>No detections yet.</span>}
          </div>

          <div className="list-card">
            <div className="list-title">Raw boxes</div>
            {result?.detections?.length ? (
              <div className="detection-list">
                {result.detections.map((detection, index) => (
                  <article key={`${detection.class_id}-${index}`} className="detection-item">
                    <div>
                      <strong>{detection.class_name}</strong>
                      <span>Class {detection.class_id}</span>
                    </div>
                    <div>
                      <strong>{detection.confidence.toFixed(2)}</strong>
                      <span>
                        {detection.box_pixels.map((value) => Number(value).toFixed(1)).join(' / ')}
                      </span>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <p className="empty-note">Upload a frame to populate class-by-class detections.</p>
            )}
          </div>
        </aside>
      </section>
    </main>
  );
}
