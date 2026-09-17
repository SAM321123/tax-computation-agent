import { useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''
const apiUrl = (path) => `${API_BASE}${path}`

const formatApiError = (payload, status) => {
  const detail = payload?.detail
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object') {
    const messages = [detail.message]
    if (Array.isArray(detail.documents)) {
      messages.push(...detail.documents.map((document) => {
        const name = document.filename || 'Input document'
        const note = document.notes || document.extraction_status || 'needs review'
        return `${name}: ${note}`
      }))
    }
    return messages.filter(Boolean).join(' ') || `Request failed (${status}).`
  }
  return payload?.message || `Request failed (${status}).`
}

const navItems = [
  { label: 'Overview', icon: 'grid' },
  { label: 'New computation', icon: 'plus' },
  { label: 'Computation history', icon: 'clock' },
]

const formatMoney = (value) =>
  new Intl.NumberFormat('en-AE', {
    style: 'currency',
    currency: 'AED',
    maximumFractionDigits: 0,
  }).format(value || 0)

const formatBytes = (bytes) => {
  if (!bytes) return '0 KB'
  return `${(bytes / 1024 / 1024).toFixed(bytes > 1024 * 1024 ? 1 : 0)} MB`
}

function Icon({ name, size = 18 }) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round' }
  const paths = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
    plus: <><path d="M12 5v14M5 12h14" /></>,
    clock: <><circle cx="12" cy="12" r="8.5" /><path d="M12 7v5l3.5 2" /></>,
    upload: <><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" /><path d="M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" /></>,
    file: <><path d="M6 3.5h8l4 4V20.5H6z" /><path d="M14 3.5v4h4M9 12h6M9 15.5h6" /></>,
    xlsx: <><path d="M6 3.5h8l4 4V20.5H6z" /><path d="M14 3.5v4h4M9 11l4 5M13 11l-4 5" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    arrow: <><path d="M5 12h14M13 6l6 6-6 6" /></>,
    search: <><circle cx="10.7" cy="10.7" r="6.2" /><path d="m16 16 4.2 4.2" /></>,
    alert: <><path d="M12 4 21 20H3z" /><path d="M12 9v5M12 17.5h.01" /></>,
    shield: <><path d="M12 3.5 19 6v5.1c0 4.7-3 7.8-7 9.4-4-1.6-7-4.7-7-9.4V6z" /><path d="m8.7 12 2.1 2.1 4.5-4.5" /></>,
    download: <><path d="M12 4v12m0 0-4-4m4 4 4-4M5 20h14" /></>,
    rotate: <><path d="M4 12a8 8 0 1 0 2.3-5.7L4 8.5" /><path d="M4 4v4.5h4.5" /></>,
    chevron: <path d="m9 18 6-6-6-6" />,
  }
  return <svg {...common} aria-hidden="true">{paths[name]}</svg>
}

function App() {
  const [activeNav, setActiveNav] = useState('New computation')
  const [inputDocs, setInputDocs] = useState([])
  const [isDragging, setIsDragging] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [options, setOptions] = useState({
    smallBusinessRelief: false,
    freeZonePerson: false,
    freeZoneRatio: 0,
    priorYearLosses: 0,
  })
  const inputRef = useRef(null)

  const addInputDocs = (files) => {
    setError('')
    const next = Array.from(files || []).filter((file) => file.size <= 25 * 1024 * 1024 && !inputDocs.some((item) => item.name === file.name && item.size === file.size))
    if (Array.from(files || []).some((file) => file.size > 25 * 1024 * 1024)) setError('Each input file must be 25 MB or smaller.')
    setInputDocs((current) => [...current, ...next])
    setResult(null)
  }

  const removeInputDoc = (fileToRemove) => {
    setInputDocs((current) => current.filter((file) => !(file.name === fileToRemove.name && file.size === fileToRemove.size)))
  }

  const runComputation = async () => {
    if (!inputDocs.length) {
      setError('Upload at least one input document before running the computation.')
      return
    }
    setIsRunning(true)
    setError('')
    try {
      const query = new URLSearchParams({
        small_business_relief: String(options.smallBusinessRelief),
        is_qualifying_free_zone_person: String(options.freeZonePerson),
        qualifying_free_zone_income_ratio: String(Number(options.freeZoneRatio) || 0),
        prior_year_tax_losses: String(Number(options.priorYearLosses) || 0),
      })
      const form = new FormData()
      inputDocs.forEach((file) => form.append('input_documents', file))
      let response
      try {
        response = await fetch(apiUrl(`/compute?${query.toString()}`), { method: 'POST', body: form })
      } catch {
        throw new Error('Unable to reach the FastAPI server on port 8000.')
      }
      const data = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(formatApiError(data, response.status))
      setResult(data)
      setActiveNav('Computation history')
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setIsRunning(false)
    }
  }

  const reset = () => {
    setInputDocs([])
    setResult(null)
    setError('')
    setActiveNav('New computation')
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">T</span><span>TaxPilot</span></div>
        <div className="workspace-switcher"><span className="workspace-avatar">A</span><span><strong>Atlas Advisory</strong><small>Tax workspace</small></span><Icon name="chevron" size={15} /></div>
        <p className="nav-label">Workspace</p>
        <nav>
          {navItems.map((item) => <button key={item.label} className={`nav-item ${activeNav === item.label ? 'active' : ''}`} onClick={() => item.label === 'New computation' ? reset() : setActiveNav(item.label)}><Icon name={item.icon} size={17} /><span>{item.label}</span></button>)}
        </nav>
        <div className="sidebar-footer"><div className="secure-line"><Icon name="shield" size={16} /><span>Secure workspace</span></div><div className="user-line"><span className="user-avatar">AM</span><span><strong>Amira Malik</strong><small>Administrator</small></span><span className="more">•••</span></div></div>
      </aside>

      <main className="main-content">
        <header className="topbar"><div className="breadcrumbs"><span>Workspace</span><Icon name="chevron" size={13} /><span className="current">{result ? 'Computation output' : 'New computation'}</span></div><div className="topbar-actions"><span className="period-pill">Tax period <strong>2025</strong></span><span className="notification-dot" /></div></header>

        {!result ? (
          <UploadView {...{ inputDocs, isDragging, setIsDragging, addInputDocs, removeInputDoc, inputRef, options, setOptions, error, isRunning, runComputation }} />
        ) : (
          <OutputView result={result} inputDocs={inputDocs} onNew={reset} />
        )}
      </main>
    </div>
  )
}

function UploadView({ inputDocs, isDragging, setIsDragging, addInputDocs, removeInputDoc, inputRef, options, setOptions, error, isRunning, runComputation }) {
  return <div className="page-wrap upload-page">
    <div className="page-heading"><div><div className="eyebrow">UAE CORPORATE TAX</div><h1>Start a new computation</h1><p>Upload your input documents and let the computation engine prepare an auditable tax position.</p></div><div className="status-chip"><span className="status-dot" /> Ready to calculate</div></div>
    <div className="stepper"><div className="step active"><span>1</span><strong>Input documents</strong></div><div className="step-line" /><div className="step"><span>2</span><strong>Tax computation</strong></div><div className="step-line" /><div className="step"><span>3</span><strong>Review output</strong></div></div>
    <section className="upload-grid">
      <div className="card upload-card">
        <div className="card-heading"><div><h2>Input documents</h2><p>Upload any combination of TBs, financial statements, working files, and source documents.</p></div><span className="required-label">At least one</span></div>
        <div className={`dropzone ${isDragging ? 'dragging' : ''} ${inputDocs.length ? 'has-file' : ''}`} onDragOver={(event) => { event.preventDefault(); setIsDragging(true) }} onDragLeave={() => setIsDragging(false)} onDrop={(event) => { event.preventDefault(); setIsDragging(false); addInputDocs(event.dataTransfer.files) }} onClick={() => inputRef.current?.click()}>
          <input ref={inputRef} type="file" multiple onChange={(event) => addInputDocs(event.target.files)} />
          <div className="upload-icon"><Icon name={inputDocs.length ? 'check' : 'upload'} size={23} /></div>
          <strong>{inputDocs.length ? `${inputDocs.length} input document${inputDocs.length > 1 ? 's' : ''} ready` : 'Drop your input documents here'}</strong><span>or <em>browse from your computer</em></span><small>TB, financial statements, working files, PDFs, DOCX, CSV, XLSX, and more · Up to 25 MB each</small>
        </div>
        {inputDocs.length > 0 && <div className="file-list">{inputDocs.map((file) => <div className="file-row" key={`${file.name}-${file.size}`}><span className="mini-file"><Icon name={file.name.toLowerCase().endsWith('.xlsx') ? 'xlsx' : 'file'} size={15} /></span><span>{file.name}<small>{formatBytes(file.size)}</small></span><button aria-label={`Remove ${file.name}`} onClick={() => removeInputDoc(file)}>×</button></div>)}</div>}
      </div>
      <div className="card options-card"><div className="card-heading"><div><h2>Computation settings</h2><p>Tell us about the entity before you run the calculation.</p></div><span className="optional-label">Optional</span></div><label className="toggle-row"><span><strong>Small Business Relief</strong><small>Apply if eligible for the relief threshold.</small></span><input type="checkbox" checked={options.smallBusinessRelief} onChange={(event) => setOptions({ ...options, smallBusinessRelief: event.target.checked })} /><i /></label><label className="toggle-row"><span><strong>Qualifying Free Zone Person</strong><small>Split qualifying income at 0% and other income at 9%.</small></span><input type="checkbox" checked={options.freeZonePerson} onChange={(event) => setOptions({ ...options, freeZonePerson: event.target.checked })} /><i /></label>{options.freeZonePerson && <label className="field"><span>Qualifying income ratio</span><div className="input-suffix"><input type="number" min="0" max="1" step="0.01" value={options.freeZoneRatio} onChange={(event) => setOptions({ ...options, freeZoneRatio: event.target.value })} /><b>ratio</b></div></label>}<label className="field"><span>Prior-year tax losses</span><div className="input-suffix"><input type="number" min="0" value={options.priorYearLosses} onChange={(event) => setOptions({ ...options, priorYearLosses: event.target.value })} /><b>AED</b></div></label></div>
    </section>
    {error && <div className="error-banner"><Icon name="alert" size={18} /><span>{error}</span></div>}
    <div className="action-bar"><div className="action-note"><Icon name="shield" size={17} /><span>AI extraction is reviewed before tax rules are applied.</span></div><button className="primary-button" disabled={!inputDocs.length || isRunning} onClick={runComputation}>{isRunning ? <><span className="spinner" /> Analyzing documents...</> : <>Analyze documents <Icon name="arrow" size={17} /></>}</button></div>
    <div className="trust-row"><span><Icon name="check" size={14} /> Format-agnostic intake</span><span><Icon name="check" size={14} /> Law-grounded reasoning</span><span><Icon name="check" size={14} /> Human review queue</span></div>
  </div>
}

function OutputView({ result, inputDocs, onNew }) {
  const reviewCount = result.review_queue?.length || 0
  return <div className="page-wrap output-page">
    <div className="page-heading output-heading"><div><div className="eyebrow">COMPUTATION COMPLETE</div><h1>Tax computation output</h1><p>Review the calculated position and the evidence trail before filing.</p></div><div className="output-actions"><button className="secondary-button" onClick={onNew}><Icon name="rotate" size={16} /> New computation</button><button className="secondary-button" onClick={() => window.print()}><Icon name="download" size={16} /> Export view</button></div></div>
    <div className="result-banner"><div className="result-symbol"><Icon name="check" size={24} /></div><div><strong>Computation ready for review</strong><p>{reviewCount ? `${reviewCount} item${reviewCount > 1 ? 's' : ''} need human review before filing.` : 'No items need additional review before filing.'}</p></div><span className="result-date">Generated just now</span></div>
    <section className="metrics-grid"><Metric label="Corporate Tax due" value={formatMoney(result.tax_due)} accent /><Metric label="Taxable income" value={formatMoney(result.taxable_income)} /><Metric label="Accounting profit" value={formatMoney(result.accounting_profit)} /><Metric label="Adjustments" value={formatMoney((result.taxable_income || 0) - (result.accounting_profit || 0))} /></section>
    <IngestionCard profiles={result.ingestion_summary || []} />
    <LineItemMappingCard items={result.line_items || []} />
    <section className="output-grid"><div className="card adjustments-card"><div className="card-heading"><div><h2>Tax adjustments</h2><p>Every adjustment is linked to an FTA law source or verified AI classification.</p></div><div className="table-tools"><span className="count-pill">{result.adjustments?.length || 0} adjustments</span><button className="icon-button"><Icon name="search" size={17} /></button></div></div><div className="table-wrap"><table><thead><tr><th>Adjustment</th><th>Amount</th><th>Evidence</th><th>Confidence</th></tr></thead><tbody>{(result.adjustments || []).map((adjustment, index) => <tr key={`${adjustment.label}-${index}`}><td><strong>{adjustment.label}</strong><small>Tax computation rule</small>{adjustment.source_sheet && <small className="source-sheet">Source sheet: {adjustment.source_sheet}</small>}</td><td className={adjustment.amount >= 0 ? 'amount-positive' : 'amount-negative'}>{adjustment.amount >= 0 ? '+' : ''}{formatMoney(adjustment.amount)}</td><td><span className={`evidence ${adjustment.citation_source === 'law_document' ? 'coded' : 'rag'}`}><span />{adjustment.citation_source === 'law_document' ? 'FTA law verified' : adjustment.citation_source === 'rag_classified' ? 'AI + law verified' : 'Unverified'}</span><small className="citation">{adjustment.citation}</small></td><td><Confidence value={adjustment.confidence} review={adjustment.needs_review} /></td></tr>)}</tbody></table></div></div><div className="side-stack"><div className="card review-card"><div className="card-heading"><div><h2>Review queue</h2><p>Items to confirm before filing.</p></div><span className={`review-count ${reviewCount ? 'warning' : 'good'}`}>{reviewCount}</span></div>{reviewCount ? result.review_queue.map((item, index) => <div className="review-item" key={`${item.label}-${index}`}><div className="review-icon"><Icon name="alert" size={15} /></div><div><strong>{item.line_ref || item.label}</strong><p>{item.notes}</p><small>{Math.round(item.confidence * 100)}% confidence</small></div></div>) : <div className="empty-review"><div><Icon name="check" size={18} /></div><strong>All clear</strong><p>No low-confidence adjustments were detected.</p></div>}</div><div className="card sources-card"><div className="card-heading"><div><h2>Input files</h2><p>Files attached to this computation run.</p></div></div>{(result.input_documents || inputDocs.map((file) => ({ filename: file.name, size_bytes: file.size }))).map((file, index) => <SourceFileMeta file={file} key={`${file.filename}-${index}`} />)}</div></div></section>
    <div className="output-footnote"><Icon name="shield" size={16} /><span>TaxPilot provides a computation aid, not a filed tax return. Review all flagged items with a qualified tax professional.</span></div>
  </div>
}

function Metric({ label, value, accent }) { return <div className={`metric-card ${accent ? 'accent' : ''}`}><span>{label}</span><strong>{value}</strong>{accent && <small><span className="metric-dot" /> Final calculated liability</small>}</div> }
function Confidence({ value, review }) { return <div className="confidence"><div><span>{Math.round((value || 0) * 100)}%</span>{review && <em>Review</em>}</div><span className="confidence-bar"><i style={{ width: `${Math.round((value || 0) * 100)}%` }} /></span></div> }
function SourceFile({ file, role }) { return <div className="source-file"><span className="mini-file"><Icon name={file?.name?.endsWith('.xlsx') ? 'xlsx' : 'file'} size={15} /></span><span><strong>{file?.name || 'Input document'}</strong><small>{role} · {formatBytes(file?.size)}</small></span><Icon name="check" size={15} /></div> }
function SourceFileMeta({ file }) { return <div className="source-file"><span className="mini-file"><Icon name={file?.filename?.endsWith('.xlsx') ? 'xlsx' : 'file'} size={15} /></span><span><strong>{file?.filename || 'Input document'}</strong><small>{file?.role || 'Input document'} · {formatBytes(file?.size_bytes)}</small></span><Icon name="check" size={15} /></div> }
function IngestionCard({ profiles }) { return <div className="card ingestion-card"><div className="card-heading"><div><h2>AI intake summary</h2><p>How the uploaded files were interpreted before computation.</p></div></div><div className="ingestion-list">{profiles.map((profile, index) => <div className="ingestion-item" key={`${profile.filename}-${index}`}><span className={`mini-file ${profile.extraction_status === 'extracted' ? '' : 'needs-review-file'}`}><Icon name={profile.extraction_status === 'extracted' ? 'check' : 'alert'} size={15} /></span><span><strong>{profile.filename}</strong><small>{profile.document_type} · {profile.detected_rows || 0} rows · {profile.extraction_status === 'extracted' ? 'Extracted' : 'Needs review'}</small></span></div>)}</div></div> }
function LineItemMappingCard({ items }) {
  const mappedCount = items.filter((item) => item.mapped_category).length
  const reviewCount = items.filter((item) => item.needs_review).length
  return <section className="card mapping-card"><div className="card-heading"><div><h2>AI line-item mapping</h2><p>Every source line is assigned a tax category before the final calculation.</p></div><span className={`mapping-summary ${mappedCount === items.length && items.length ? 'complete' : 'partial'}`}>{mappedCount}/{items.length} mapped{reviewCount ? ` · ${reviewCount} review` : ''}</span></div><div className="table-wrap"><table><thead><tr><th>Source line item</th><th>Amount</th><th>Mapped category</th><th>Tax treatment</th><th>Evidence</th><th>Confidence</th></tr></thead><tbody>{items.map((item, index) => <tr key={`${item.account_name}-${index}`}><td><strong>{item.account_name}</strong><small>{item.source_file ? `${item.source_file}${item.source_sheet ? ` · ${item.source_sheet}` : ''}${item.source_row ? ` · row ${item.source_row}` : ''}` : 'Input document'}</small></td><td className="line-amount">{formatMoney(item.amount)}</td><td><span className={`mapping-status ${item.mapping_status === 'mapped' ? 'mapped' : 'review'}`}>{item.mapping_status === 'mapped' ? 'Mapped' : 'Review required'}</span><small>{item.mapped_category.replaceAll('_', ' ')}</small></td><td>{item.tax_treatment}</td><td><span className={`evidence ${item.mapping_method === 'ai_verified' ? 'rag' : item.mapping_method === 'law_document' ? 'coded' : 'rag'}`}><span />{item.mapping_method === 'ai_verified' ? 'AI + law verified' : item.mapping_method === 'law_document' ? 'FTA law verified' : 'AI review'}</span><small className="citation">{item.citation || 'No adjustment citation'}</small></td><td><Confidence value={item.confidence} review={item.needs_review} /></td></tr>)}</tbody></table></div></section>
}

export default App
