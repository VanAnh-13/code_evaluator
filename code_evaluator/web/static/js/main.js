/**
 * Code Analyzer - Dashboard JavaScript
 * Handles CodeMirror editor, AJAX analysis, theme toggle, drag-and-drop
 */

(function () {
    'use strict';

    /* ================================================================
       Language ↔ CodeMirror mode mapping
       ================================================================ */
    const LANG_MODES = {
        cpp: 'text/x-c++src',
        c: 'text/x-csrc',
        python: 'python',
        javascript: 'javascript',
        java: 'text/x-java',
        go: 'text/x-go',
        rust: 'text/x-rustsrc',
        ruby: 'text/x-ruby',
        php: 'text/x-php',
        swift: 'text/x-swift',
        html: 'htmlmixed',
        css: 'css',
        auto: 'text/x-c++src'
    };

    const EXT_LANG = {
        c: 'c', h: 'c', cpp: 'cpp', cxx: 'cpp', cc: 'cpp', hpp: 'cpp',
        py: 'python', js: 'javascript', mjs: 'javascript', ts: 'javascript',
        java: 'java', go: 'go', rs: 'rust', rb: 'ruby',
        php: 'php', swift: 'swift', html: 'html', css: 'css'
    };

    /* ================================================================
       State
       ================================================================ */
    let editor = null;   // CodeMirror instance
    let currentLang = 'auto';

    /* ================================================================
       DOM Ready
       ================================================================ */
    document.addEventListener('DOMContentLoaded', function () {
        initTheme();
        initFlashMessages();
        initCodeMirror();
        initFileUpload();
        initDragDrop();
        initAnalyzeButton();
        initTabs();
        initScoreRings();
        initAgentMode();
    });

    /* ================================================================
       Theme Toggle
       ================================================================ */
    function initTheme() {
        const saved = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', saved);
        updateEditorTheme(saved);

        const btn = document.getElementById('theme-toggle');
        if (!btn) return;
        updateThemeIcon(btn, saved);

        btn.addEventListener('click', function () {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('theme', next);
            updateThemeIcon(btn, next);
            updateEditorTheme(next);
        });
    }

    function updateThemeIcon(btn, theme) {
        const icon = btn.querySelector('i');
        if (!icon) return;
        icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }

    function updateEditorTheme(theme) {
        if (!editor) return;
        editor.setOption('theme', theme === 'dark' ? 'dracula' : 'eclipse');
    }

    /* ================================================================
       Flash Messages
       ================================================================ */
    function initFlashMessages() {
        document.querySelectorAll('.flash-message').forEach(function (msg) {
            const closeBtn = msg.querySelector('.close-btn');
            if (closeBtn) {
                closeBtn.addEventListener('click', function () { msg.remove(); });
            }
            setTimeout(function () { msg.style.opacity = '0'; setTimeout(function () { msg.remove(); }, 300); }, 5000);
        });
    }

    /* ================================================================
       CodeMirror Editor
       ================================================================ */
    function initCodeMirror() {
        const textarea = document.getElementById('code-editor');
        if (!textarea) return;

        const theme = document.documentElement.getAttribute('data-theme') || 'dark';

        editor = CodeMirror.fromTextArea(textarea, {
            mode: LANG_MODES['cpp'],
            theme: theme === 'dark' ? 'dracula' : 'eclipse',
            lineNumbers: true,
            matchBrackets: true,
            autoCloseBrackets: true,
            styleActiveLine: true,
            tabSize: 4,
            indentWithTabs: false,
            lineWrapping: false,
            viewportMargin: Infinity
        });

        // Status bar updates
        editor.on('cursorActivity', updateEditorStatus);
        editor.on('change', updateEditorStatus);

        // Language selector
        const langSelect = document.getElementById('language-select');
        if (langSelect) {
            langSelect.addEventListener('change', function () {
                currentLang = this.value;
                const mode = LANG_MODES[currentLang] || LANG_MODES['auto'];
                editor.setOption('mode', mode);
                updateEditorStatus();
            });
        }

        updateEditorStatus();
    }

    function updateEditorStatus() {
        if (!editor) return;
        const cursor = editor.getCursor();
        const lineEl = document.getElementById('status-line');
        const charEl = document.getElementById('status-char');
        const langEl = document.getElementById('status-lang');

        if (lineEl) lineEl.textContent = 'Ln ' + (cursor.line + 1);
        if (charEl) charEl.textContent = 'Col ' + (cursor.ch + 1);
        if (langEl) {
            const langSelect = document.getElementById('language-select');
            const lang = langSelect ? langSelect.value : 'auto';
            langEl.textContent = lang === 'auto' ? 'Auto Detect' : lang.toUpperCase();
        }
    }

    /* ================================================================
       File Upload (into editor)
       ================================================================ */
    function initFileUpload() {
        const uploadBtn = document.getElementById('file-upload-btn');
        const fileInput = document.getElementById('file-input');
        if (!uploadBtn || !fileInput) return;

        uploadBtn.addEventListener('click', function () { fileInput.click(); });

        fileInput.addEventListener('change', function () {
            if (this.files.length === 0) return;
            readFileIntoEditor(this.files[0]);
            this.value = '';
        });
    }

    function readFileIntoEditor(file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            if (editor) {
                editor.setValue(e.target.result);
                detectLanguage(file.name);
            }
        };
        reader.readAsText(file);
    }

    function detectLanguage(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const lang = EXT_LANG[ext] || 'auto';
        const langSelect = document.getElementById('language-select');
        if (langSelect) {
            langSelect.value = lang;
            langSelect.dispatchEvent(new Event('change'));
        }
    }

    /* ================================================================
       Drag and Drop
       ================================================================ */
    function initDragDrop() {
        const wrapper = document.querySelector('.editor-wrapper');
        const overlay = document.querySelector('.drop-overlay');
        if (!wrapper || !overlay) return;

        ['dragenter', 'dragover'].forEach(function (evt) {
            wrapper.addEventListener(evt, function (e) {
                e.preventDefault();
                e.stopPropagation();
                overlay.classList.add('active');
            });
        });

        ['dragleave', 'drop'].forEach(function (evt) {
            wrapper.addEventListener(evt, function (e) {
                e.preventDefault();
                e.stopPropagation();
                overlay.classList.remove('active');
            });
        });

        wrapper.addEventListener('drop', function (e) {
            const files = e.dataTransfer.files;
            if (files.length > 0) readFileIntoEditor(files[0]);
        });
    }

    /* ================================================================
       Analyze Button → AJAX
       ================================================================ */
    function initAnalyzeButton() {
        const btn = document.getElementById('analyze-btn');
        if (!btn) return;

        btn.addEventListener('click', function (e) {
            e.preventDefault();
            if (!editor) return;

            const code = editor.getValue().trim();
            if (!code) {
                showToast('Please enter or upload code to analyze.', 'warning');
                return;
            }

            const langSelect = document.getElementById('language-select');
            const language = langSelect ? langSelect.value : 'auto';

            // Route to agent or standard analysis
            if (agentEnabled) {
                runAgentAnalysis(code, language);
                return;
            }

            const csrfToken = document.querySelector('input[name="csrf_token"]');

            // Show loading state
            showPanel('loading');
            btn.disabled = true;

            const headers = { 'Content-Type': 'application/json' };
            if (csrfToken) headers['X-CSRFToken'] = csrfToken.value;

            fetch('/api/analyze', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({ code: code, language: language })
            })
                .then(function (resp) {
                    if (!resp.ok) throw new Error('Server error: ' + resp.status);
                    return resp.json();
                })
                .then(function (data) {
                    if (data.error) throw new Error(data.error);
                    renderResults(data);
                    showPanel('results');
                })
                .catch(function (err) {
                    renderError(err.message);
                    showPanel('results');
                })
                .finally(function () { btn.disabled = false; });
        });
    }

    function showPanel(which) {
        var empty = document.getElementById('empty-state');
        var loading = document.getElementById('loading-state');
        var agentLoading = document.getElementById('agent-loading-state');
        var results = document.getElementById('analysis-results');
        if (empty) empty.style.display = (which === 'empty') ? '' : 'none';
        if (loading) loading.style.display = (which === 'loading') ? '' : 'none';
        if (agentLoading) agentLoading.style.display = (which === 'agent-loading') ? '' : 'none';
        if (results) results.style.display = (which === 'results') ? '' : 'none';
    }

    /* ================================================================
       Render AJAX Results
       ================================================================ */
    function renderResults(data) {
        var container = document.getElementById('analysis-results');
        if (!container) return;

        var score = data.overall_score || 0;
        var summary = data.summary || 'Analysis complete.';
        var issues = data.issues || [];
        var fixes = data.suggested_fixes || [];
        var lang = data.language || '';

        // Determine score color
        var scoreColor = score >= 80 ? '#3fb950' : (score >= 50 ? '#d29922' : '#f85149');
        var circumference = 2 * Math.PI * 42;
        var offset = circumference - (score / 100) * circumference;

        var html = '';

        // Score + summary row
        html += '<div class="dashboard-row">';
        html += '<div class="score-card">';
        html += '  <div class="score-ring-inline"><svg viewBox="0 0 100 100" width="100" height="100">';
        html += '    <circle class="score-bg" cx="50" cy="50" r="42"/>';
        html += '    <circle class="score-fill" cx="50" cy="50" r="42" stroke="' + scoreColor + '" stroke-dasharray="' + circumference + '" stroke-dashoffset="' + offset + '"/>';
        html += '  </svg><div class="score-text" style="color:' + scoreColor + '">' + score + '</div></div>';
        html += '  <div class="score-label">Overall Score</div>';
        html += '</div>';
        html += '<div class="summary-card"><h3><i class="fas fa-file-alt"></i> Summary</h3>';
        html += '<p class="summary-text">' + escapeHtml(summary) + '</p>';
        html += renderIssueCounts(issues);
        html += '</div></div>';

        // Issues
        if (issues.length > 0) {
            html += '<div class="issues-section">';
            html += renderTabs(issues);
            html += '<div class="issues-list" id="ajax-issues-list">';
            html += renderIssueCards(issues);
            html += '</div></div>';
        }

        // Suggested fixes
        if (fixes.length > 0) {
            html += '<div class="fixes-section"><h3><i class="fas fa-magic"></i> Suggested Fixes</h3>';
            fixes.forEach(function (fix) {
                html += '<div class="fix-card">';
                html += '<div class="fix-header">Line ' + (fix.line || '?') + ' — ' + escapeHtml(fix.explanation || '') + '</div>';
                html += '<pre class="fix-code">' + escapeHtml(fix.original || '') + '\n→\n' + escapeHtml(fix.fixed || '') + '</pre>';
                html += '</div>';
            });
            html += '</div>';
        }

        container.innerHTML = html;
        initTabs();      // re-bind tab clicks within results
    }

    function renderIssueCounts(issues) {
        var counts = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
        issues.forEach(function (i) { var s = (i.severity || 'info').toLowerCase(); if (counts[s] !== undefined) counts[s]++; });
        var html = '<div class="issue-counts">';
        html += '<div class="count-item"><span class="count-number text-danger">' + (counts.critical + counts.high) + '</span><span class="count-label">Critical/High</span></div>';
        html += '<div class="count-item"><span class="count-number text-warning">' + counts.medium + '</span><span class="count-label">Medium</span></div>';
        html += '<div class="count-item"><span class="count-number">' + counts.low + '</span><span class="count-label">Low</span></div>';
        html += '<div class="count-item"><span class="count-number">' + counts.info + '</span><span class="count-label">Info</span></div>';
        html += '</div>';
        return html;
    }

    function renderTabs(issues) {
        var categories = ['all'];
        var catCounts = {};
        issues.forEach(function (i) {
            var c = (i.category || 'other').toLowerCase();
            if (categories.indexOf(c) === -1) categories.push(c);
            catCounts[c] = (catCounts[c] || 0) + 1;
        });
        catCounts['all'] = issues.length;

        var html = '<div class="tabs">';
        categories.forEach(function (cat, idx) {
            html += '<button class="tab' + (idx === 0 ? ' active' : '') + '" data-category="' + cat + '">';
            html += cat.charAt(0).toUpperCase() + cat.slice(1);
            html += ' <span class="tab-count">' + catCounts[cat] + '</span>';
            html += '</button>';
        });
        html += '</div>';
        return html;
    }

    function renderIssueCards(issues) {
        var html = '';
        issues.forEach(function (issue) {
            var sev = (issue.severity || 'info').toLowerCase();
            var cat = (issue.category || 'other').toLowerCase();
            html += '<div class="issue-card-js sev-' + sev + '" data-category="' + cat + '">';
            html += '<div class="issue-meta-js">';
            if (issue.line) html += '<span class="issue-line"><i class="fas fa-map-marker-alt"></i> Line ' + issue.line + '</span>';
            html += '<span class="severity-badge severity-' + sev + '">' + sev + '</span>';
            html += '<span class="category-badge">' + cat + '</span>';
            html += '</div>';
            html += '<div class="issue-desc-js">' + escapeHtml(issue.description || '') + '</div>';
            if (issue.recommendation) {
                html += '<div class="issue-rec-js"><i class="fas fa-lightbulb"></i><span>' + escapeHtml(issue.recommendation) + '</span></div>';
            }
            html += '</div>';
        });
        return html;
    }

    function renderError(message) {
        var container = document.getElementById('analysis-results');
        if (!container) return;
        container.innerHTML = '<div class="error-card"><i class="fas fa-exclamation-triangle"></i><h3>Analysis Failed</h3><p>' + escapeHtml(message) + '</p></div>';
    }

    /* ================================================================
       Tabs (Category filter)
       ================================================================ */
    function initTabs() {
        document.querySelectorAll('.tabs').forEach(function (tabBar) {
            tabBar.querySelectorAll('.tab').forEach(function (tab) {
                tab.addEventListener('click', function () {
                    tabBar.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('active'); });
                    tab.classList.add('active');
                    var cat = tab.getAttribute('data-category');
                    var list = tabBar.parentElement.querySelector('.issues-list');
                    if (!list) return;
                    list.querySelectorAll('[data-category]').forEach(function (card) {
                        card.style.display = (cat === 'all' || card.getAttribute('data-category') === cat) ? '' : 'none';
                    });
                });
            });
        });
    }

    /* ================================================================
       Score Ring Animation (Server-rendered pages)
       ================================================================ */
    function initScoreRings() {
        document.querySelectorAll('.score-ring[data-score]').forEach(function (ring) {
            var score = parseInt(ring.getAttribute('data-score'), 10) || 0;
            var fill = ring.querySelector('.score-fill');
            if (!fill) return;
            var circumference = 2 * Math.PI * 42;
            var offset = circumference - (score / 100) * circumference;
            fill.style.strokeDasharray = circumference;
            fill.style.strokeDashoffset = offset;

            if (score >= 80) fill.style.stroke = '#3fb950';
            else if (score >= 50) fill.style.stroke = '#d29922';
            else fill.style.stroke = '#f85149';
        });
    }

    /* ================================================================
       Helpers
       ================================================================ */
    function escapeHtml(str) {
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function showToast(message, type) {
        var container = document.querySelector('.flash-messages');
        if (!container) {
            container = document.createElement('div');
            container.className = 'flash-messages';
            document.body.appendChild(container);
        }
        var msg = document.createElement('div');
        msg.className = 'flash-message ' + (type || 'info');
        msg.innerHTML = '<span>' + escapeHtml(message) + '</span><button class="close-btn">&times;</button>';
        container.appendChild(msg);
        msg.querySelector('.close-btn').addEventListener('click', function () { msg.remove(); });
        setTimeout(function () { msg.remove(); }, 5000);
    }

    /* ================================================================
       Agent Mode
       ================================================================ */
    var agentEnabled = false;
    var agentSessionId = null;
    var agentEventSource = null;

    function initAgentMode() {
        var toggle = document.getElementById('agent-mode-toggle');
        if (!toggle) return;

        toggle.addEventListener('change', function () {
            agentEnabled = this.checked;
            var btn = document.getElementById('analyze-btn');
            if (btn) {
                var icon = btn.querySelector('i');
                var text = btn.querySelector('span');
                if (agentEnabled) {
                    icon.className = 'fas fa-robot';
                    text.textContent = 'Agent Analyze';
                } else {
                    icon.className = 'fas fa-magnifying-glass-chart';
                    text.textContent = 'Analyze Code';
                }
            }
        });
    }

    function runAgentAnalysis(code, language) {
        var btn = document.getElementById('analyze-btn');
        btn.disabled = true;
        showPanel('agent-loading');

        var timeline = document.getElementById('agent-steps-timeline');
        if (timeline) timeline.innerHTML = '';

        // Step 1: Create session
        fetch('/api/agent/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ max_steps: 15 })
        })
        .then(function (resp) { return resp.json(); })
        .then(function (data) {
            agentSessionId = data.session_id;

            // Step 2: Send code for analysis
            return fetch('/api/agent/sessions/' + agentSessionId + '/messages', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    code: code,
                    language: language,
                    message: 'Please analyze this ' + language + ' code thoroughly.'
                })
            });
        })
        .then(function (resp) { return resp.json(); })
        .then(function () {
            // Step 3: Open SSE stream
            agentEventSource = new EventSource('/api/agent/sessions/' + agentSessionId + '/stream');

            agentEventSource.onmessage = function (event) {
                var step = JSON.parse(event.data);

                if (step.type === 'session_end') {
                    agentEventSource.close();
                    agentEventSource = null;
                    onAgentComplete(agentSessionId);
                    return;
                }

                renderAgentStep(step);
            };

            agentEventSource.onerror = function () {
                agentEventSource.close();
                agentEventSource = null;
                onAgentComplete(agentSessionId);
            };
        })
        .catch(function (err) {
            renderError('Agent error: ' + err.message);
            showPanel('results');
            btn.disabled = false;
        });
    }

    function renderAgentStep(step) {
        var timeline = document.getElementById('agent-steps-timeline');
        if (!timeline) return;

        var el = document.createElement('div');
        el.className = 'agent-step agent-step-' + step.type;

        var icons = {
            thinking: 'fa-brain',
            tool_call: 'fa-screwdriver-wrench',
            tool_result: 'fa-clipboard-check',
            response: 'fa-comment-dots',
            error: 'fa-triangle-exclamation'
        };
        var icon = icons[step.type] || 'fa-circle';

        var html = '<div class="step-icon"><i class="fas ' + icon + '"></i></div>';
        html += '<div class="step-content">';
        html += '<div class="step-header">';
        html += '<span class="step-type">' + step.type.replace('_', ' ') + '</span>';
        html += '<span class="step-number">#' + step.step_number + '</span>';
        html += '</div>';

        if (step.type === 'tool_call' && step.tool_name) {
            html += '<div class="step-tool-name">' + escapeHtml(step.tool_name) + '</div>';
            if (step.tool_args && Object.keys(step.tool_args).length > 0) {
                var argsStr = Object.keys(step.tool_args).map(function(k) {
                    var v = String(step.tool_args[k]);
                    return k + '=' + (v.length > 50 ? v.substring(0, 50) + '...' : v);
                }).join(', ');
                html += '<div class="step-args">' + escapeHtml(argsStr) + '</div>';
            }
        } else if (step.type === 'tool_result' && step.tool_result) {
            var truncResult = step.tool_result.length > 200
                ? step.tool_result.substring(0, 200) + '...'
                : step.tool_result;
            html += '<pre class="step-result">' + escapeHtml(truncResult) + '</pre>';
        } else if (step.content) {
            html += '<div class="step-text">' + escapeHtml(step.content.substring(0, 300)) + '</div>';
        }

        html += '</div>';
        el.innerHTML = html;
        timeline.appendChild(el);
        timeline.scrollTop = timeline.scrollHeight;
    }

    function onAgentComplete(sessionId) {
        // Fetch full session results
        fetch('/api/agent/sessions/' + sessionId)
        .then(function (resp) { return resp.json(); })
        .then(function (session) {
            var btn = document.getElementById('analyze-btn');
            if (btn) btn.disabled = false;

            if (session.result) {
                // Render as standard results if the agent returned structured data
                var result = session.result;
                if (result.issues || result.overall_score !== undefined) {
                    renderResults({
                        language: result.language || '',
                        summary: result.summary || result.full_response || '',
                        overall_score: result.overall_score || 0,
                        issues: result.issues || [],
                        suggested_fixes: result.suggested_fixes || []
                    });

                    // Append agent steps summary
                    var container = document.getElementById('analysis-results');
                    if (container) {
                        var agentInfo = '<div class="agent-summary-card">';
                        agentInfo += '<h4><i class="fas fa-robot"></i> Agent Analysis Details</h4>';
                        agentInfo += '<div class="agent-stats">';
                        agentInfo += '<span>Steps: ' + session.step_count + '</span>';
                        agentInfo += '<span>Files analyzed: ' + (session.context.files_analyzed || []).length + '</span>';
                        agentInfo += '<span>Issues found: ' + (session.context.issues_found || 0) + '</span>';
                        agentInfo += '<span>Fixes applied: ' + (session.context.fixes_applied || 0) + '</span>';
                        agentInfo += '<span>Fixes verified: ' + (session.context.fixes_verified || 0) + '</span>';
                        agentInfo += '</div></div>';
                        container.innerHTML += agentInfo;
                    }
                    showPanel('results');
                } else {
                    // Show raw agent response
                    var container = document.getElementById('analysis-results');
                    if (container) {
                        container.innerHTML = '<div class="agent-response-card">'
                            + '<h3><i class="fas fa-robot"></i> Agent Response</h3>'
                            + '<pre class="agent-raw-response">' + escapeHtml(result.summary || result.full_response || JSON.stringify(result, null, 2)) + '</pre>'
                            + '</div>';
                    }
                    showPanel('results');
                }
            } else if (session.error) {
                renderError('Agent error: ' + session.error);
                showPanel('results');
            } else {
                renderError('Agent finished without producing results.');
                showPanel('results');
            }
        })
        .catch(function (err) {
            var btn = document.getElementById('analyze-btn');
            if (btn) btn.disabled = false;
            renderError('Failed to retrieve agent results: ' + err.message);
            showPanel('results');
        });
    }

})();
