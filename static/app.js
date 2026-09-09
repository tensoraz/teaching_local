let studentId = localStorage.getItem('studentId') || '';
let currentSection = 'content';
let isTeacher = false;
let selectedAnswer = null;
let isEditMode = false;

const $ = id => document.getElementById(id);

async function api(path, body) {
    try {
        const options = body ? {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        } : { cache: 'no-store' };
        const r = await fetch(path, options);
        if (!r.ok) return { ok: false };
        return await r.json();
    } catch (e) {
        console.error('API error:', e);
        return { ok: false };
    }
}

function escapeHtml(value) {
    return String(value || '').replace(/[&<>"']/g, function(c) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c];
    });
}

function markdownToHtml(md) {
    if (!md) return '<p style="color:#999;">No content yet.</p>';
    var html = md;
    html = html.replace(/```(\w+)\n([\s\S]*?)```/g, function(m, lang, code) {
        return '<div style="background:#1e1e1e;color:#d4d4d4;padding:15px;border-radius:8px;margin:10px 0;font-family:monospace;"><pre>' + escapeHtml(code) + '</pre></div>';
    });
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
    html = html.replace(/\n/g, '<br>');
    return html;
}

function showMsg(msg, ok) {
    if (ok === undefined) ok = true;
    var div = document.createElement('div');
    div.style.cssText = 'position:fixed;top:20px;right:20px;padding:15px 20px;border-radius:8px;color:white;z-index:9999;background:' + (ok ? '#4CAF50' : '#f44336') + ';font-size:16px;';
    div.textContent = msg;
    document.body.appendChild(div);
    setTimeout(function() { div.remove(); }, 3000);
}

// ============ TAB SWITCHING ============
function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
    document.querySelectorAll('.tab').forEach(function(btn) {
        var onclick = btn.getAttribute('onclick') || '';
        if (onclick.indexOf(tabName) !== -1) btn.classList.add('active');
    });
    ['teach', 'topics', 'add', 'students', 'announce'].forEach(function(t) {
        var el = $(t + 'Tab');
        if (el) el.style.display = (t === tabName) ? 'block' : 'none';
    });
    if (tabName === 'topics') renderTopics();
    if (tabName === 'students') renderStudents();
}

// ============ SECTION SWITCHING ============
// FIX: also tell the server which section is active so /api/state stays in sync
async function switchSection(section) {
    currentSection = section;
    isEditMode = false;

    document.querySelectorAll('.section-btn').forEach(function(b) { b.classList.remove('active'); });
    document.querySelectorAll('.section-btn').forEach(function(btn) {
        var onclick = btn.getAttribute('onclick') || '';
        if (onclick.indexOf(section) !== -1) btn.classList.add('active');
    });

    var viewMode = $('viewMode');
    var editMode = $('editMode');
    if (viewMode) viewMode.style.display = 'block';
    if (editMode) editMode.style.display = 'none';

    // Keep server in sync
    var state = await api('/api/state');
    if (state.topics && state.topics.length) {
        await api('/api/navigate', { topic: state.current_topic, section: section });
    }

    refreshTeacher();
}

// ============ NAVIGATION ============
async function navigateTopic(dir) {
    var state = await api('/api/state');
    if (!state.topics || !state.topics.length) return;
    var newIndex = state.current_topic + dir;
    if (newIndex < 0 || newIndex >= state.topics.length) return;
    await api('/api/navigate', { topic: newIndex, section: currentSection });
    isEditMode = false;
    var viewMode = $('viewMode');
    var editMode = $('editMode');
    if (viewMode) viewMode.style.display = 'block';
    if (editMode) editMode.style.display = 'none';
    refreshTeacher();
}

// ============ ADD TOPIC ============
async function addTopic() {
    var titleInput = $('newTopicTitle');
    var descInput  = $('newTopicDesc');
    if (!titleInput || !titleInput.value.trim()) {
        showMsg('Enter topic title', false);
        return;
    }
    var result = await api('/api/teacher/add_topic', {
        title: titleInput.value.trim(),
        description: descInput ? descInput.value.trim() : ''
    });
    if (result.ok) {
        showMsg('Topic added! Now add content, quiz, and assignment');
        titleInput.value = '';
        if (descInput) descInput.value = '';
        switchTab('teach');
        refreshTeacher();
    }
}

// ============ RENDER TOPICS ============
async function renderTopics() {
    var state = await api('/api/state');
    var list = $('topicsList');
    if (!list) return;
    if (!state.topics || !state.topics.length) {
        list.innerHTML = '<p style="color:#666;text-align:center;padding:40px;">No topics yet. Go to "Add Topic" tab.</p>';
        return;
    }
    list.innerHTML = state.topics.map(function(topic, i) {
        return '<div style="border:2px solid ' + (i === state.current_topic ? '#4CAF50' : '#ddd') + ';padding:15px;margin:10px 0;border-radius:8px;cursor:pointer;" onclick="selectTopic(' + i + ')">' +
            '<h3>' + escapeHtml(topic.title) + '</h3>' +
            '<p style="color:#666;">' + escapeHtml(topic.description || '') + '</p>' +
            '<div style="margin-top:10px;">' +
                '<span style="font-size:12px;background:' + (topic.sections.content.body ? '#4CAF50' : '#aaa') + ';color:white;padding:3px 8px;border-radius:4px;margin-right:5px;">Content ' + (topic.sections.content.body ? '✓' : '✗') + '</span>' +
                '<span style="font-size:12px;background:' + (topic.sections.quiz.question ? '#4CAF50' : '#aaa') + ';color:white;padding:3px 8px;border-radius:4px;margin-right:5px;">Quiz ' + (topic.sections.quiz.question ? '✓' : '✗') + '</span>' +
                '<span style="font-size:12px;background:' + (topic.sections.assignment.description ? '#4CAF50' : '#aaa') + ';color:white;padding:3px 8px;border-radius:4px;">Assignment ' + (topic.sections.assignment.description ? '✓' : '✗') + '</span>' +
            '</div>' +
            '<button onclick="event.stopPropagation();deleteTopic(\'' + topic.id + '\')" style="background:#f44336;color:white;border:none;padding:8px 15px;border-radius:4px;cursor:pointer;margin-top:10px;">Delete</button>' +
        '</div>';
    }).join('');
}

async function selectTopic(index) {
    await api('/api/navigate', { topic: index, section: currentSection });
    isEditMode = false;
    var viewMode = $('viewMode');
    var editMode = $('editMode');
    if (viewMode) viewMode.style.display = 'block';
    if (editMode) editMode.style.display = 'none';
    switchTab('teach');
    refreshTeacher();
}

async function deleteTopic(id) {
    if (!confirm('Delete this topic?')) return;
    await api('/api/teacher/delete_topic', { topic_id: id });
    renderTopics();
    refreshTeacher();
}

// ============ STUDENTS ============
async function renderStudents() {
    var state = await api('/api/state');
    var list = $('studentsList');
    if (!list) return;
    list.innerHTML = (state.students && state.students.length) ?
        state.students.map(function(s) { return '<p style="padding:10px;border-bottom:1px solid #eee;">👤 ' + escapeHtml(s.name) + '</p>'; }).join('') :
        '<p>No students connected</p>';
}

// ============ ANNOUNCEMENTS ============
async function sendAnnouncement() {
    var el = $('announcementText');
    var msg = el ? el.value.trim() : '';
    if (!msg) return;
    await api('/api/teacher/announce', { message: msg });
    if (el) el.value = '';
    showMsg('Announcement sent!');
}

// ============ EDIT MODE ============
function showEditMode() {
    isEditMode = true;
    var viewMode = $('viewMode');
    var editMode = $('editMode');
    if (viewMode) viewMode.style.display = 'none';
    if (editMode) editMode.style.display = 'block';
    renderEditArea();
}

function cancelEdit() {
    isEditMode = false;
    var viewMode = $('viewMode');
    var editMode = $('editMode');
    if (viewMode) viewMode.style.display = 'block';
    if (editMode) editMode.style.display = 'none';
    refreshTeacher();
}

async function renderEditArea() {
    var state = await api('/api/state');
    if (!state.topics || !state.topics.length) {
        showMsg('No topic selected. Add a topic first.', false);
        return;
    }
    var topic = state.topics[state.current_topic];
    if (!topic) return;

    // FIX: always use currentSection (local JS var) to pick the section data
    var section = topic.sections[currentSection];
    if (!section) return;

    var editArea = $('editArea');
    if (!editArea) return;

    if (currentSection === 'content') {
        editArea.innerHTML =
            '<div class="edit-box">' +
                '<h3>✏️ Edit Content</h3>' +
                '<p style="color:#666;margin:10px 0;">Write your content using Markdown</p>' +
                '<textarea id="edit_content_body" style="width:100%;min-height:400px;padding:15px;font-family:monospace;font-size:14px;border:2px solid #ff9800;border-radius:8px;">' + escapeHtml(section.body || '') + '</textarea>' +
            '</div>';

    } else if (currentSection === 'quiz') {
        var options = (section.options && section.options.length) ? section.options : ['', ''];
        var correctVal = (typeof section.correct === 'number') ? section.correct : 0;
        var rows = options.map(function(opt, i) {
            return '<div style="display:flex;gap:10px;margin:10px 0;align-items:center;">' +
                '<input type="radio" name="edit_correct" value="' + i + '" ' + (i === correctVal ? 'checked' : '') + ' style="width:20px;height:20px;cursor:pointer;">' +
                '<input type="text" class="edit-option-input" value="' + escapeHtml(opt) + '" placeholder="Option ' + (i + 1) + '" style="flex:1;padding:12px;border:1px solid #ddd;border-radius:6px;">' +
            '</div>';
        }).join('');

        editArea.innerHTML =
            '<div class="edit-box">' +
                '<h3>✏️ Edit Quiz</h3>' +
                '<label style="display:block;font-weight:bold;margin-top:15px;">Question:</label>' +
                '<textarea id="edit_quiz_question" style="width:100%;min-height:100px;padding:12px;border:1px solid #ddd;border-radius:6px;font-size:16px;">' + escapeHtml(section.question || '') + '</textarea>' +
                '<label style="display:block;font-weight:bold;margin-top:20px;">Options (select correct answer with the radio button):</label>' +
                '<div id="editOptions">' + rows + '</div>' +
                '<button onclick="addOption()" class="btn btn-blue" style="margin-top:8px;">+ Add Option</button>' +
                '<label style="display:block;font-weight:bold;margin-top:20px;">Explanation (shown after answer):</label>' +
                '<textarea id="edit_quiz_explanation" style="width:100%;min-height:80px;padding:12px;border:1px solid #ddd;border-radius:6px;">' + escapeHtml(section.explanation || '') + '</textarea>' +
                '<div style="display:flex;gap:20px;margin-top:20px;">' +
                    '<div><label style="font-weight:bold;">Points:</label>' +
                    '<input type="number" id="edit_quiz_points" value="' + (section.points || 10) + '" min="1" style="width:100px;padding:10px;border:1px solid #ddd;border-radius:6px;margin-left:5px;"></div>' +
                    '<div><label style="font-weight:bold;">Time (seconds):</label>' +
                    '<input type="number" id="edit_quiz_time" value="' + (section.time_limit || 60) + '" min="10" style="width:100px;padding:10px;border:1px solid #ddd;border-radius:6px;margin-left:5px;"></div>' +
                '</div>' +
            '</div>';

    } else if (currentSection === 'assignment') {
        var tasks = (section.tasks && section.tasks.length) ? section.tasks : [''];
        var taskInputs = tasks.map(function(task) {
            return '<input type="text" class="edit-task-input" value="' + escapeHtml(task) + '" placeholder="Enter task" style="width:100%;padding:12px;margin:5px 0;border:1px solid #ddd;border-radius:6px;font-size:16px;">';
        }).join('');

        editArea.innerHTML =
            '<div class="edit-box">' +
                '<h3>✏️ Edit Assignment</h3>' +
                '<label style="display:block;font-weight:bold;margin-top:15px;">Title:</label>' +
                '<input type="text" id="edit_assignment_title" value="' + escapeHtml(section.title || '') + '" style="width:100%;padding:12px;margin:10px 0;border:1px solid #ddd;border-radius:6px;font-size:16px;">' +
                '<label style="display:block;font-weight:bold;margin-top:10px;">Description:</label>' +
                '<textarea id="edit_assignment_desc" style="width:100%;min-height:100px;padding:12px;border:1px solid #ddd;border-radius:6px;">' + escapeHtml(section.description || '') + '</textarea>' +
                '<label style="display:block;font-weight:bold;margin-top:20px;">Tasks:</label>' +
                '<div id="editTasks">' + taskInputs + '</div>' +
                '<button onclick="addTask()" class="btn btn-blue" style="margin-top:8px;">+ Add Task</button>' +
                '<div style="display:flex;gap:20px;margin-top:20px;">' +
                    '<div><label style="font-weight:bold;">Due Date:</label>' +
                    '<input type="date" id="edit_assignment_due" value="' + escapeHtml(section.due_date || '') + '" style="padding:10px;border:1px solid #ddd;border-radius:6px;margin-left:5px;"></div>' +
                    '<div><label style="font-weight:bold;">Max Points:</label>' +
                    '<input type="number" id="edit_assignment_points" value="' + (section.max_points || 100) + '" min="1" style="width:100px;padding:10px;border:1px solid #ddd;border-radius:6px;margin-left:5px;"></div>' +
                '</div>' +
                '<label style="display:block;font-weight:bold;margin-top:20px;">Instructions:</label>' +
                '<textarea id="edit_assignment_instructions" style="width:100%;min-height:100px;padding:12px;border:1px solid #ddd;border-radius:6px;">' + escapeHtml(section.instructions || '') + '</textarea>' +
            '</div>';
    }
}

async function saveSection() {
    var state = await api('/api/state');
    if (!state.topics || !state.topics.length) {
        showMsg('No topic selected', false);
        return;
    }
    var topic = state.topics[state.current_topic];
    if (!topic) return;

    if (currentSection === 'content') {
        var bodyEl = $('edit_content_body');
        if (bodyEl) {
            var result = await api('/api/teacher/update_content', {
                topic_id: topic.id,
                content: { body: bodyEl.value }
            });
            if (result.ok) {
                showMsg('✅ Content saved!');
                cancelEdit();
            } else {
                showMsg('Save failed', false);
            }
        }

    } else if (currentSection === 'quiz') {
        var questionEl = $('edit_quiz_question');
        if (!questionEl || !questionEl.value.trim()) {
            showMsg('Please enter a question', false);
            return;
        }
        var options = [];
        document.querySelectorAll('.edit-option-input').forEach(function(inp) {
            if (inp.value.trim()) options.push(inp.value.trim());
        });
        if (options.length < 2) {
            showMsg('Add at least 2 options', false);
            return;
        }
        var correctRadio = document.querySelector('input[name="edit_correct"]:checked');
        var correctIndex = correctRadio ? parseInt(correctRadio.value) : 0;

        var result = await api('/api/teacher/update_quiz', {
            topic_id: topic.id,
            quiz: {
                question:    questionEl.value.trim(),
                options:     options,
                correct:     correctIndex,
                explanation: $('edit_quiz_explanation')  ? $('edit_quiz_explanation').value.trim()       : '',
                points:      $('edit_quiz_points')       ? (parseInt($('edit_quiz_points').value)  || 10) : 10,
                time_limit:  $('edit_quiz_time')         ? (parseInt($('edit_quiz_time').value)    || 60) : 60
            }
        });
        if (result.ok) {
            showMsg('✅ Quiz saved!');
            cancelEdit();
        } else {
            showMsg('Save failed', false);
        }

    } else if (currentSection === 'assignment') {
        var descEl = $('edit_assignment_desc');
        if (!descEl || !descEl.value.trim()) {
            showMsg('Please enter a description', false);
            return;
        }
        var tasks = [];
        document.querySelectorAll('.edit-task-input').forEach(function(inp) {
            if (inp.value.trim()) tasks.push(inp.value.trim());
        });

        var result = await api('/api/teacher/update_assignment', {
            topic_id: topic.id,
            assignment: {
                title:        $('edit_assignment_title')        ? $('edit_assignment_title').value.trim()                   : '',
                description:  descEl.value.trim(),
                tasks:        tasks,
                due_date:     $('edit_assignment_due')          ? $('edit_assignment_due').value                            : '',
                max_points:   $('edit_assignment_points')       ? (parseInt($('edit_assignment_points').value) || 100)      : 100,
                instructions: $('edit_assignment_instructions') ? $('edit_assignment_instructions').value.trim()            : ''
            }
        });
        if (result.ok) {
            showMsg('✅ Assignment saved!');
            cancelEdit();
        } else {
            showMsg('Save failed', false);
        }
    }
}

function addOption() {
    var div = $('editOptions');
    if (!div) return;
    var count = div.children.length;
    var newRow = document.createElement('div');
    newRow.style.cssText = 'display:flex;gap:10px;margin:10px 0;align-items:center;';
    newRow.innerHTML =
        '<input type="radio" name="edit_correct" value="' + count + '" style="width:20px;height:20px;cursor:pointer;">' +
        '<input type="text" class="edit-option-input" placeholder="Option ' + (count + 1) + '" style="flex:1;padding:12px;border:1px solid #ddd;border-radius:6px;">';
    div.appendChild(newRow);
}

function addTask() {
    var div = $('editTasks');
    if (!div) return;
    var input = document.createElement('input');
    input.type = 'text';
    input.className = 'edit-task-input';
    input.placeholder = 'Enter task';
    input.style.cssText = 'width:100%;padding:12px;margin:5px 0;border:1px solid #ddd;border-radius:6px;font-size:16px;';
    div.appendChild(input);
}

// ============ REFRESH TEACHER ============
// FIX: use currentSection (local JS var) for BOTH section lookup AND render branch
async function refreshTeacher() {
    var state = await api('/api/state');
    if (!state || !state.topics || !state.topics.length) {
        var titleEl  = $('currentTopicTitle');
        var descEl   = $('currentTopicDesc');
        var countEl  = $('studentCount');
        var contentEl = $('teachContent');
        if (titleEl)   titleEl.textContent  = 'No topics yet';
        if (descEl)    descEl.textContent   = 'Go to "Add Topic" tab to create your first topic';
        if (countEl)   countEl.textContent  = '0 students';
        if (contentEl) contentEl.innerHTML  = '<p style="color:#666;text-align:center;padding:40px;">No content. Add a topic first.</p>';
        return;
    }

    var topic = state.topics[state.current_topic];
    if (!topic) return;

    var titleEl   = $('currentTopicTitle');
    var descEl    = $('currentTopicDesc');
    var countEl   = $('studentCount');
    var contentEl = $('teachContent');

    if (titleEl)  titleEl.textContent  = topic.title;
    if (descEl)   descEl.textContent   = topic.description || '';
    if (countEl)  countEl.textContent  = (state.student_count || 0) + ' students';

    if (!contentEl) return;

    // FIX: use currentSection for the lookup — not state.current_section
    var section = topic.sections[currentSection];

    if (currentSection === 'content') {
        contentEl.innerHTML = markdownToHtml(section ? section.body : '');

    } else if (currentSection === 'quiz') {
        if (section && section.question) {
            var optHtml = (section.options || []).map(function(opt, i) {
                return '<p style="padding:12px;margin:8px 0;background:' + (i === section.correct ? '#e8f5e9' : '#f5f5f5') + ';border-radius:6px;font-size:16px;">' +
                    (i === section.correct ? '✅' : '○') + ' ' + escapeHtml(opt) + '</p>';
            }).join('');
            contentEl.innerHTML =
                '<h3>📝 Quiz</h3>' +
                '<div style="background:#e3f2fd;padding:10px;border-radius:6px;margin-bottom:15px;"><strong>Students can see this quiz</strong></div>' +
                '<p style="margin:15px 0;font-size:18px;"><strong>Question:</strong> ' + escapeHtml(section.question) + '</p>' +
                '<div style="margin:15px 0;">' + optHtml + '</div>' +
                '<p><strong>Points:</strong> ' + (section.points || 10) + '</p>' +
                (section.explanation ? '<p><strong>Explanation:</strong> ' + escapeHtml(section.explanation) + '</p>' : '');
        } else {
            contentEl.innerHTML = '<p style="color:#666;padding:20px;">No quiz yet. Click "Edit This Section" to add.</p>';
        }

    } else if (currentSection === 'assignment') {
        if (section && section.description) {
            var taskHtml = (section.tasks || []).map(function(t) { return '<li style="margin:5px 0;">' + escapeHtml(t) + '</li>'; }).join('');
            contentEl.innerHTML =
                '<h3>📚 Assignment</h3>' +
                '<div style="background:#e3f2fd;padding:10px;border-radius:6px;margin-bottom:15px;"><strong>Students can see this assignment</strong></div>' +
                '<p><strong>Title:</strong> ' + escapeHtml(section.title || topic.title) + '</p>' +
                '<p style="margin:15px 0;font-size:16px;">' + escapeHtml(section.description) + '</p>' +
                '<h4>Tasks:</h4>' +
                '<ol style="margin:10px 0 10px 20px;">' + taskHtml + '</ol>' +
                (section.due_date   ? '<p><strong>Due:</strong> '        + section.due_date   + '</p>' : '') +
                (section.max_points ? '<p><strong>Max Points:</strong> ' + section.max_points + '</p>' : '');
        } else {
            contentEl.innerHTML = '<p style="color:#666;padding:20px;">No assignment yet. Click "Edit This Section" to add.</p>';
        }
    }
}

// ============ STUDENT FUNCTIONS ============
async function joinClass() {
    var nameEl = $('studentName');
    var name = nameEl ? nameEl.value.trim() : '';
    if (!name) {
        var msgEl = $('joinMsg');
        if (msgEl) msgEl.textContent = 'Enter your name';
        return;
    }
    var result = await api('/api/join', { name: name });
    if (result.ok) {
        studentId = result.id;
        localStorage.setItem('studentId', studentId);
        var joinView = $('joinView');
        var classView = $('classView');
        if (joinView) joinView.style.display = 'none';
        if (classView) classView.style.display = 'block';
        refreshStudent();
    }
}

function studentSwitchSection(section) {
    currentSection = section;
    document.querySelectorAll('.section-btn').forEach(function(b) { b.classList.remove('active'); });
    document.querySelectorAll('.section-btn').forEach(function(btn) {
        var onclick = btn.getAttribute('onclick') || '';
        if (onclick.indexOf(section) !== -1) btn.classList.add('active');
    });
    refreshStudent();
}

async function refreshStudent() {
    var state = await api('/api/state');
    if (!state || !state.topics || !state.topics.length) {
        var t = $('currentTopicTitle');
        var c = $('studentCount');
        var sc = $('studentContent');
        if (t) t.textContent = 'Waiting for teacher...';
        if (c) c.textContent = (state.student_count || 0) + ' students';
        if (sc) sc.innerHTML = '<p style="color:#666;text-align:center;padding:40px;">Waiting for teacher to add topics...</p>';
        return;
    }

    var topic = state.topics[state.current_topic];
    if (!topic) return;

    var titleEl = $('currentTopicTitle');
    var countEl = $('studentCount');
    var content = $('studentContent');

    if (titleEl) titleEl.textContent = topic.title;
    if (countEl) countEl.textContent = (state.student_count || 0) + ' students';

    if (!content) return;

    // Student always follows the server's current_section (teacher drives it)
    var activeSection = state.current_section;
    var section = topic.sections[activeSection];

    if (activeSection === 'content') {
        content.innerHTML = (section && section.body) ? markdownToHtml(section.body) : '<p>No content yet</p>';

    } else if (activeSection === 'quiz') {
        if (section && section.question) {
            var optHtml = (section.options || []).map(function(opt, i) {
                return '<div onclick="selectAnswer(' + i + ')" id="opt_' + i + '" style="padding:12px;margin:8px 0;border:2px solid #ddd;border-radius:6px;cursor:pointer;font-size:16px;">' + escapeHtml(opt) + '</div>';
            }).join('');
            content.innerHTML =
                '<h3>📝 Quiz</h3>' +
                '<p style="font-size:18px;margin:15px 0;"><strong>Question:</strong> ' + escapeHtml(section.question) + '</p>' +
                '<div style="margin:15px 0;">' + optHtml + '</div>' +
                '<button onclick="submitQuiz()" style="padding:12px 24px;background:#4CAF50;color:white;border:none;border-radius:6px;cursor:pointer;font-size:16px;">Submit</button>';
        } else {
            content.innerHTML = '<p>No quiz yet</p>';
        }

    } else if (activeSection === 'assignment') {
        if (section && section.description) {
            var taskHtml = (section.tasks || []).map(function(t) { return '<li>' + escapeHtml(t) + '</li>'; }).join('');
            content.innerHTML =
                '<h3>📚 Assignment</h3>' +
                '<p><strong>Title:</strong> ' + escapeHtml(section.title || topic.title) + '</p>' +
                '<p style="margin:10px 0;">' + escapeHtml(section.description) + '</p>' +
                '<h4>Tasks:</h4><ol>' + taskHtml + '</ol>' +
                '<textarea id="submission" style="width:100%;min-height:150px;margin-top:15px;padding:12px;border:1px solid #ddd;border-radius:6px;font-family:monospace;" placeholder="Write your answer or code here..."></textarea>' +
                '<button onclick="submitAssignment()" style="margin-top:10px;padding:12px 24px;background:#2196F3;color:white;border:none;border-radius:6px;cursor:pointer;font-size:16px;">Submit</button>';
        } else {
            content.innerHTML = '<p>No assignment yet</p>';
        }
    }

    // Show announcements if any
    if (state.announcements && state.announcements.length) {
        var latest = state.announcements[state.announcements.length - 1];
        var annEl = $('announcementBanner');
        if (annEl) {
            annEl.textContent = '📢 ' + latest.message;
            annEl.style.display = 'block';
        }
    }
}

function selectAnswer(index) {
    selectedAnswer = index;
    document.querySelectorAll('[id^="opt_"]').forEach(function(el, i) {
        el.style.background = (i === index) ? '#4CAF50' : 'white';
        el.style.color      = (i === index) ? 'white'   : 'black';
        el.style.border     = (i === index) ? '2px solid #4CAF50' : '2px solid #ddd';
    });
}

async function submitQuiz() {
    if (selectedAnswer === null) {
        showMsg('Select an answer first', false);
        return;
    }
    var result = await api('/api/quiz/submit', { id: studentId, answer: selectedAnswer });
    if (result.ok) {
        showMsg(result.correct ? ('✅ Correct! +' + result.points + ' points') : '❌ Incorrect. Try again next time.');
        selectedAnswer = null;
    }
}

async function submitAssignment() {
    var el = $('submission');
    var submission = el ? el.value.trim() : '';
    if (!submission) {
        showMsg('Write your submission first', false);
        return;
    }
    var result = await api('/api/assignment/submit', { id: studentId, submission: submission });
    if (result.ok) showMsg('✅ Submitted successfully!');
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    isTeacher = !!$('teachTab');
    if (isTeacher) {
        refreshTeacher();
        setInterval(function() {
            var active = document.activeElement;
            if (active && (active.tagName === 'TEXTAREA' || active.tagName === 'INPUT')) return;
            if (!isEditMode) refreshTeacher();
        }, 3000);
    } else {
        if (studentId) {
            var joinView = $('joinView');
            var classView = $('classView');
            if (joinView) joinView.style.display = 'none';
            if (classView) classView.style.display = 'block';
            refreshStudent();
        }
        setInterval(refreshStudent, 3000);
    }
});