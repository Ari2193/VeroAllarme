import { useState, useEffect } from 'react'
import { alertsAPI, statsAPI } from './api/client'
import './App.css'

function App() {
  const [stats, setStats] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      setLoading(true)
      const [statsResponse, alertsResponse] = await Promise.all([
        statsAPI.getStats(),
        alertsAPI.getAlerts()
      ])
      
      setStats(statsResponse.data)
      setAlerts(alertsResponse.data.alerts)
      setError(null)
    } catch (err) {
      console.error('Failed to fetch data:', err)
      setError('Failed to connect to backend')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>🔒 VeroAllarme</h1>
        <p>AI-Powered Smart Motion Alert Filtering</p>
      </header>

      <main className="container">
        {loading && <div className="loading">Loading...</div>}
        
        {error && (
          <div className="error">
            <p>⚠️ {error}</p>
            <button onClick={fetchData}>Retry</button>
          </div>
        )}

        {!loading && !error && (
          <>
            <section className="stats">
              <h2>📊 Dashboard Statistics</h2>
              <div className="stats-grid">
                <div className="stat-card">
                  <h3>Total Alerts</h3>
                  <p className="stat-value">{stats?.total_alerts || 0}</p>
                </div>
                <div className="stat-card">
                  <h3>False Positives</h3>
                  <p className="stat-value">{stats?.false_positives || 0}</p>
                </div>
                <div className="stat-card">
                  <h3>YOLO Invocations</h3>
                  <p className="stat-value">{stats?.yolo_invocations || 0}</p>
                </div>
                <div className="stat-card">
                  <h3>Accuracy</h3>
                  <p className="stat-value">{stats?.accuracy || 0}%</p>
                </div>
              </div>
            </section>

            <section className="alerts">
              <h2>🚨 Recent Alerts</h2>
              {alerts.length === 0 ? (
                <p className="empty-state">No alerts yet. System is ready to receive alerts from cameras.</p>
              ) : (
                <div className="alerts-list">
                  {alerts.map((alert) => (
                    <div key={alert.id} className="alert-card">
                      <h3>{alert.title}</h3>
                      <p>{alert.description}</p>
                    </div>
                  ))}
                </div>
              )}
            </section>

            <section className="quick-start">
              <h2>🚀 Quick Start</h2>
              <ol>
                <li>Configure your security camera webhook to POST to <code>/api/alerts</code></li>
                <li>Define masked regions using the mask editor</li>
                <li>Start receiving and filtering alerts intelligently</li>
                <li>Provide feedback to improve accuracy over time</li>
              </ol>
            </section>
          </>
        )}
      </main>

      <footer>
        <p>Built for Hackathon Excellence 🏆</p>
      </footer>
    </div>
  )
}

export default App
