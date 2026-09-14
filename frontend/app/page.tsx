'use client'

import { useState } from 'react'

type Source = {
  customer_query: string
  agent_response: string
  similarity: number
}

type ChatResponse = {
  query: string
  answer: string | null
  decision: string
  escalate: boolean
  reason: string
  risk_level: string
  risk_categories: string[]
  confidence: number
  confidence_level: string
  max_similarity: number | null
  average_similarity: number | null
  retrieved_results: Source[]
}

export default function Home() {
  const [query, setQuery] = useState('')
  const [response, setResponse] = useState<ChatResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const sendMessage = async () => {
    if (!query.trim()) return

    setLoading(true)
    setError('')
    setResponse(null)

    try {
      const res = await fetch('http://127.0.0.1:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          top_k: 5,
        }),
      })

      if (!res.ok) {
        throw new Error('Failed to get response from SupportIQ API')
      }

      const data: ChatResponse = await res.json()
      setResponse(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <div className="mx-auto max-w-6xl px-6 py-10">
        {/* Header */}
        <header className="mb-10">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">SupportIQ</h1>
              <p className="mt-2 text-slate-400">
                AI-powered customer support intelligence
              </p>
            </div>

            <div className="flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-sm">
              <span className="h-2.5 w-2.5 rounded-full bg-green-400" />
              API Online
            </div>
          </div>
        </header>

        {/* Main grid */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Chat section */}
          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6 lg:col-span-2">
            <h2 className="mb-5 text-xl font-semibold">Customer Support</h2>

            <div className="mb-4 min-h-40 rounded-xl border border-slate-800 bg-slate-950 p-5">
              {!response && !loading && (
                <div className="flex h-32 items-center justify-center text-center text-slate-500">
                  Ask a customer-support question to begin.
                </div>
              )}

              {loading && (
                <div className="flex h-32 items-center justify-center">
                  <div className="text-slate-400">
                    Searching support knowledge...
                  </div>
                </div>
              )}

              {response && (
                <div className="space-y-5">
                  {/* Customer */}
                  <div>
                    <p className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-500">
                      Customer
                    </p>

                    <div className="rounded-xl bg-slate-800 p-4">
                      {response.query}
                    </div>
                  </div>

                  {/* Decision */}
                  <div className="flex flex-wrap gap-2">
                    <span
                      className={`rounded-full px-3 py-1 text-xs font-semibold ${
                        response.decision === 'ai'
                          ? 'bg-green-500/10 text-green-400'
                          : 'bg-red-500/10 text-red-400'
                      }`}
                    >
                      {response.decision === 'ai'
                        ? 'AI HANDLED'
                        : 'HUMAN ESCALATION'}
                    </span>

                    <span className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-300">
                      Confidence {(response.confidence * 100).toFixed(1)}%
                    </span>

                    <span className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-300">
                      Risk: {response.risk_level}
                    </span>
                  </div>

                  {/* Answer */}
                  <div>
                    <p className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-500">
                      SupportIQ
                    </p>

                    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 leading-7 text-slate-200">
                      {response.answer ||
                        'This conversation should be handled by a human support agent.'}
                    </div>
                  </div>

                  {/* Reason */}
                  <div className="rounded-xl bg-slate-800/50 p-4">
                    <p className="text-xs uppercase tracking-wider text-slate-500">
                      Decision Reason
                    </p>

                    <p className="mt-1 text-sm text-slate-300">
                      {response.reason}
                    </p>

                    {response.risk_categories.length > 0 && (
                      <p className="mt-2 text-sm text-red-400">
                        Risk categories: {response.risk_categories.join(', ')}
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Input */}
            <div className="flex gap-3">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    sendMessage()
                  }
                }}
                placeholder="Ask a customer support question..."
                className="flex-1 rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 outline-none transition focus:border-slate-500"
              />

              <button
                onClick={sendMessage}
                disabled={loading || !query.trim()}
                className="rounded-xl bg-white px-6 py-3 font-semibold text-slate-950 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loading ? '...' : 'Send'}
              </button>
            </div>

            {error && (
              <div className="mt-4 rounded-xl border border-red-900 bg-red-950/30 p-4 text-sm text-red-400">
                {error}
              </div>
            )}
          </section>

          {/* Analytics */}
          <aside className="space-y-6">
            {/* Confidence */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
              <h3 className="font-semibold">Retrieval Confidence</h3>

              {response ? (
                <div className="mt-5">
                  <div className="text-4xl font-bold">
                    {(response.confidence * 100).toFixed(1)}%
                  </div>

                  <p className="mt-2 text-sm text-slate-400">
                    {response.confidence_level} confidence
                  </p>

                  <div className="mt-5 h-2 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-white"
                      style={{
                        width: `${response.confidence * 100}%`,
                      }}
                    />
                  </div>
                </div>
              ) : (
                <p className="mt-4 text-sm text-slate-500">
                  No query processed yet.
                </p>
              )}
            </div>

            {/* Retrieved sources */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
              <h3 className="font-semibold">Retrieved Sources</h3>

              {response ? (
                <div className="mt-4 space-y-3">
                  {response.retrieved_results.map((source, index) => (
                    <div key={index} className="rounded-xl bg-slate-950 p-3">
                      <div className="flex justify-between gap-3">
                        <span className="text-xs text-slate-500">
                          Source {index + 1}
                        </span>

                        <span className="text-xs font-medium text-slate-300">
                          {(source.similarity * 100).toFixed(1)}%
                        </span>
                      </div>

                      <p className="mt-2 line-clamp-3 text-sm text-slate-400">
                        {source.customer_query}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mt-4 text-sm text-slate-500">
                  Retrieved support examples will appear here.
                </p>
              )}
            </div>
          </aside>
        </div>
      </div>
    </main>
  )
}
