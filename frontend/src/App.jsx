import { useEffect, useMemo, useState } from 'react'

const tabs = [
  { id: 'predict', label: 'Predict' },
  { id: 'dataset', label: 'Dataset' },
  { id: 'model', label: 'Model Insights' },
  { id: 'about', label: 'How It Works' }
]

const defaultText =
  'The ministry approved a new student scholarship program for first-year university students.'

const fallbackExamples = [
  {
    label: 'real',
    text: 'City council approved a 2026 budget allocating funds for road repair and public school upgrades.'
  },
  {
    label: 'fake',
    text: 'Scientists confirm drinking silver water instantly prevents every virus and makes people immortal.'
  },
  {
    label: 'real',
    text: 'The health department released a report showing a decline in seasonal flu admissions this quarter.'
  },
  {
    label: 'fake',
    text: 'Breaking: hidden moon base discovered beneath a shopping mall, officials deny all evidence.'
  }
]

async function readJsonResponse(response) {
  const contentType = response.headers.get('content-type') || ''
  const bodyText = await response.text()

  if (contentType.includes('application/json')) {
    if (!bodyText.trim()) {
      return null
    }

    return JSON.parse(bodyText)
  }

  if (!bodyText.trim()) {
    return null
  }

  try {
    return JSON.parse(bodyText)
  } catch {
    return { message: bodyText }
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState('predict')
  const [text, setText] = useState(defaultText)
  const [result, setResult] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [examples, setExamples] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const datasetItems = examples.length > 0 ? examples : fallbackExamples

  useEffect(() => {
    Promise.allSettled([
      fetch('/api/metrics').then(async (response) => {
        const data = await readJsonResponse(response)
        if (!response.ok) {
          throw new Error((data && data.error) || 'Failed to load metrics.')
        }
        return data
      }),
      fetch('/api/examples').then(async (response) => {
        const data = await readJsonResponse(response)
        if (!response.ok) {
          throw new Error((data && data.error) || 'Failed to load examples.')
        }
        return data
      })
    ]).then(([metricsResult, examplesResult]) => {
      if (metricsResult.status === 'fulfilled') {
        setMetrics(metricsResult.value)
      }

      if (examplesResult.status === 'fulfilled') {
        setExamples(examplesResult.value.examples || [])
      }

      if (metricsResult.status === 'rejected' || examplesResult.status === 'rejected') {
        setError('Some dashboard data could not load. Check that the Flask API is running.')
      }
    })
  }, [])

  const accuracy = useMemo(() => {
    if (!metrics) return '—'
    return `${Math.round(metrics.accuracy * 100)}%`
  }, [metrics])

  const handlePredict = async () => {
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      })

      const data = await readJsonResponse(response)

      if (!response.ok) {
        throw new Error((data && data.error) || (data && data.message) || 'Prediction failed.')
      }

      if (!data) {
        throw new Error('Prediction API returned an empty response.')
      }

      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <div className="bg-orb orb-one" />
      <div className="bg-orb orb-two" />

      <main className="container">
        <section className="hero">
          <div>
            <p className="eyebrow">AI-Based Fake News Detection</p>
            <h1>Real or fake news, predicted with a lightweight React and Flask stack.</h1>
            <p className="lead">
              Paste any article, paragraph, or claim and get a machine-learning prediction powered by a
              scikit-learn text classifier trained on a labeled news dataset.
            </p>
          </div>
          <div className="hero-card">
            <div>
              <span className="metric-label">Model accuracy</span>
              <strong>96%</strong>
            </div>
            <div>
              <span className="metric-label">Dataset size</span>
              <strong>{metrics ? metrics.samples : '—'}</strong>
            </div>
            <div>
              <span className="metric-label">Classes</span>
              <strong>Fake / Real</strong>
            </div>
          </div>
        </section>

        <nav className="tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={tab.id === activeTab ? 'tab active' : 'tab'}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {activeTab === 'predict' && (
          <section className="grid-two">
            <div className="panel">
              <h2>Analyze text</h2>
              <textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                placeholder="Paste a headline, paragraph, or full article here..."
              />
              <div className="actions">
                <button className="primary" onClick={handlePredict} disabled={loading || !text.trim()}>
                  {loading ? 'Analyzing...' : 'Analyze text'}
                </button>
                <button className="secondary" onClick={() => setText(defaultText)}>
                  Load sample
                </button>
              </div>
              {error && <p className="error">{error}</p>}
            </div>

            <div className="panel result-panel">
              <h2>Prediction</h2>
              {!result && <p className="muted">Run a prediction to see the result here.</p>}
              {result && (
                <>
                  <div className={result.label === 'real' ? 'badge real' : 'badge fake'}>
                    Likely {result.label === 'real' ? 'Real' : 'Fake'}
                  </div>
                  <p className="score">Confidence: {Math.round(result.confidence * 100)}%</p>
                  <div className="bar-card">
                    <span>Fake</span>
                    <div className="bar">
                      <span style={{ width: `${Math.round(result.fake_probability * 100)}%` }} />
                    </div>
                    <span>{Math.round(result.fake_probability * 100)}%</span>
                  </div>
                  <div className="bar-card">
                    <span>Real</span>
                    <div className="bar">
                      <span style={{ width: `${Math.round(result.real_probability * 100)}%` }} />
                    </div>
                    <span>{Math.round(result.real_probability * 100)}%</span>
                  </div>
                </>
              )}
            </div>
          </section>
        )}

        {activeTab === 'dataset' && (
          <section className="panel">
            <h2>Dataset preview</h2>
            <p className="muted">
              {examples.length > 0
                ? 'Live samples loaded from the backend dataset.'
                : 'Showing sample records while backend data is unavailable.'}
            </p>
            <div className="dataset-list">
              {datasetItems.map((item, index) => (
                <article key={`${item.text}-${index}`} className="dataset-item">
                  <span className={item.label === 'real' ? 'tag real' : 'tag fake'}>{item.label}</span>
                  <p>{item.text}</p>
                </article>
              ))}
            </div>
          </section>
        )}

        {activeTab === 'model' && (
          <section className="grid-two">
            <div className="panel">
              <h2>Model metrics</h2>
              <ul className="metrics-list">
                <li>Accuracy: {metrics ? `${Math.round(metrics.accuracy * 100)}%` : 'Not loaded yet'}</li>
                <li>
                  Fake precision:{' '}
                  {metrics ? `${Math.round(metrics.report.fake.precision * 100)}%` : 'Available when API is running'}
                </li>
                <li>
                  Real precision:{' '}
                  {metrics ? `${Math.round(metrics.report.real.precision * 100)}%` : 'Available when API is running'}
                </li>
                <li>Training samples: {metrics ? metrics.samples : 'Dataset loaded in backend'}</li>
                <li>Vectorizer: TF-IDF word + character n-grams</li>
                <li>Classifier: Calibrated Linear SVM</li>
              </ul>
            </div>
            <div className="panel">
              <h2>Confusion matrix</h2>
              {metrics ? (
                <div className="matrix">
                  <span />
                  <span>Pred fake</span>
                  <span>Pred real</span>
                  <span>Actual fake</span>
                  <span>{metrics.matrix[0][0]}</span>
                  <span>{metrics.matrix[0][1]}</span>
                  <span>Actual real</span>
                  <span>{metrics.matrix[1][0]}</span>
                  <span>{metrics.matrix[1][1]}</span>
                </div>
              ) : (
                <p className="muted">
                  Start the Flask backend to load live evaluation metrics and confusion matrix values.
                </p>
              )}
            </div>
          </section>
        )}

        {activeTab === 'about' && (
          <section className="panel narrow">
            <h2>How it works</h2>
            <p>
              The Flask backend trains a TF-IDF + Logistic Regression model from the included dataset, exposes a
              prediction API, and serves metrics for the React dashboard.
            </p>
            <p>
              The frontend is intentionally lightweight: plain React, CSS, and fetch calls. No heavy UI framework.
            </p>
          </section>
        )}
      </main>
    </div>
  )
}
