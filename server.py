import http.server, socketserver, json, socket, urllib.parse, threading
from pathlib import Path
import uuid
import time
import mimetypes
import random
import re
import os
import secrets
import http.cookies

PORT = 8000
ROOT = Path(__file__).parent
STATIC = ROOT / 'static'
COURSES = ROOT / 'courses'          # courses/<course>/<unit>/... (one folder per unit)
RESULTS = ROOT / 'quiz_results'     # graded unit-quiz submissions, one JSON file per unit
LOCK = threading.Lock()
UNIT_RE = re.compile(r'^[a-z0-9-]+/[a-z0-9-]+$')
mimetypes.add_type('text/javascript', '.js')
mimetypes.add_type('text/csv', '.csv')
mimetypes.add_type('application/x-ipynb+json', '.ipynb')

# TEACHER ACCESS: the laptop running this server is always the teacher. Any other device
# (e.g. a smart board with its own computer) logs in once with a PIN:
#   - each course has its own PIN in courses/<course>/_pin.txt (created on first start; edit to change);
#     it unlocks only that course's teacher pages
#   - the master PIN (printed at startup, or set TEACHER_PIN) unlocks every course and /teacher
TEACHER_PIN = os.environ.get('TEACHER_PIN') or str(secrets.randbelow(900000) + 100000)
ALL = '*'                                                      # scope of the master PIN
SESSIONS = {}                                                  # cookie token -> set of scopes (course names or ALL)
TEACHER_FILES = {'board.html', 'session.html', 'results.html'}  # teacher-only pages in any course unit
TEACHER_POSTS = {'/api/navigate', '/api/reset'}                # plus everything under /api/teacher/
MAX_TRIES, LOCKOUT = 5, 600                                    # wrong PINs allowed, then seconds locked
FAILED = {}                                                    # ip -> (wrong tries, time of first try)
OWN_IPS = set()

# QUIZ LOCK: once a student starts a unit quiz, their device (IP) can open only that quiz page
# (plus shared CSS/JS) until they submit, or until the quiz time + QUIZ_GRACE runs out.
# This stops "open a new tab and read the study guide". Teacher devices are never locked.
QUIZ_LOCKS = {}                                                # ip -> {'unit': ..., 'until': epoch seconds}
QUIZ_GRACE = 90

LOCKED_PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Quiz in progress</title>
<link rel="stylesheet" href="/courses/shared/theme.css"></head><body>
<div class="card" style="max-width:520px;margin:60px auto;text-align:center">
<h2>&#128274; Quiz in progress</h2>
<p>You started the quiz, so other pages are closed until you press <b>Submit</b>.</p>
<p><a class="btn" href="/courses/{unit}/quiz.html">Back to my quiz</a></p>
</div></body></html>'''
STATE = {
    'topics': [],  # EMPTY - Teacher adds everything
    'current_topic': 0,
    'current_section': 'content',
    'students': {},
    'assignments': {},
    'quizzes_results': {},
    'teacher_name': 'Professor',
    'class_name': 'C Programming',
    'announcements': []
}

def generate_id():
    return uuid.uuid4().hex[:8]

def ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        return s.getsockname()[0]
    except:
        return '127.0.0.1'
    finally:
        s.close()

def own_ips():
    """Addresses of this laptop. Refreshed on demand because Wi-Fi can hand out a new one."""
    ips = {ip_address()}
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return ips

def new_pin():
    return str(secrets.randbelow(900000) + 100000)

def course_names():
    return sorted(d.name for d in COURSES.iterdir()
                  if d.is_dir() and (d / 'index.html').is_file() and not d.name.startswith(('_', '.')))

def course_pin(course):
    """The course's own PIN, created on first use. The '_' prefix keeps the file from being served."""
    f = COURSES / course / '_pin.txt'
    if not f.is_file():
        f.write_text(new_pin() + '\n', encoding='utf-8')
    return f.read_text(encoding='utf-8').strip()

def scope_of(url_path):
    """Which PIN a page or API needs: the course name for /courses/<course>/..., else the master scope."""
    parts = urllib.parse.unquote(urllib.parse.urlparse(url_path).path).split('/')
    if len(parts) > 2 and parts[1] == 'courses' and parts[2] in course_names():
        return parts[2]
    return ALL

def safe_next(target):
    """Only allow redirects back into this site (no '//evil.com')."""
    return target if target.startswith('/') and not target.startswith('//') else '/teacher'

def json_response(handler, obj, status=200):
    try:
        data = json.dumps(obj).encode()
        handler.send_response(status)
        handler.send_header('Content-Type', 'application/json; charset=utf-8')
        handler.send_header('Cache-Control', 'no-store')
        handler.send_header('Content-Length', str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
    except:
        pass

def course_path(url_path):
    """Map /courses/... to a path under COURSES. Names starting with '_' or '.' stay private."""
    rel = urllib.parse.unquote(url_path[len('/courses'):]).strip('/')
    parts = [x for x in rel.split('/') if x]
    if any(x.startswith(('_', '.')) for x in parts):
        return None
    try:
        target = (COURSES.joinpath(*parts)).resolve()
        target.relative_to(COURSES.resolve())
    except (ValueError, OSError):
        return None
    return target

def load_unit_quiz(unit):
    if not UNIT_RE.match(unit or ''):
        return None
    f = COURSES / unit / '_quiz.json'
    if not f.is_file():
        return None
    return json.loads(f.read_text(encoding='utf-8'))

def results_file(unit):
    return RESULTS / (unit.replace('/', '__') + '.json')

def read_results(unit):
    f = results_file(unit)
    if not f.is_file():
        return []
    try:
        return json.loads(f.read_text(encoding='utf-8'))
    except ValueError:
        return []

def grade_question(q, given):
    """Return the fraction of marks earned (0..1) for one answer."""
    kind = q.get('type', 'mcq')
    try:
        if kind == 'multi':
            if not isinstance(given, list):
                return 0.0
            return 1.0 if sorted(set(int(x) for x in given)) == sorted(q['answer']) else 0.0
        if kind == 'order':
            if not isinstance(given, list) or len(given) != len(q['answer']):
                return 0.0
            hits = sum(1 for a, b in zip(given, q['answer']) if int(a) == b)
            return hits / len(q['answer'])
        return 1.0 if given is not None and int(given) == q['answer'] else 0.0
    except (TypeError, ValueError):
        return 0.0

class Handler(http.server.BaseHTTPRequestHandler):
    def send_file(self, path, content_type):
        try:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except:
            pass

    def is_teacher(self, scope=ALL):
        """True on this laptop, or with a login cookie covering this course (or the master scope)."""
        ip = self.client_address[0]
        if ip.startswith('127.') or ip == '::1':
            return True
        if ip not in OWN_IPS:
            OWN_IPS.update(own_ips())
        if ip in OWN_IPS:
            return True
        cookie = http.cookies.SimpleCookie()
        try:
            cookie.load(self.headers.get('Cookie', ''))
        except http.cookies.CookieError:
            return False
        scopes = SESSIONS.get(cookie['teacher'].value, ()) if 'teacher' in cookie else ()
        return ALL in scopes or scope in scopes

    def session_token(self):
        cookie = http.cookies.SimpleCookie()
        try:
            cookie.load(self.headers.get('Cookie', ''))
        except http.cookies.CookieError:
            return None
        token = cookie['teacher'].value if 'teacher' in cookie else None
        return token if token in SESSIONS else None

    def redirect(self, location, cookie=None):
        self.send_response(302)
        self.send_header('Location', location)
        if cookie:
            self.send_header('Set-Cookie', cookie)
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

    def to_login(self):
        self.redirect('/login?next=' + urllib.parse.quote(self.path))

    def quiz_lock(self):
        """The unit this device is locked into, or None. Expired locks are dropped."""
        ip = self.client_address[0]
        with LOCK:
            lock = QUIZ_LOCKS.get(ip)
            if lock and lock['until'] < time.time():
                QUIZ_LOCKS.pop(ip, None)
                lock = None
        return lock['unit'] if lock else None

    def blocked_by_quiz(self, p):
        """Serve the 'quiz in progress' page and return True if this GET is not allowed during a quiz."""
        unit = self.quiz_lock()
        if not unit or self.is_teacher():
            return False
        if p in ('/courses/' + unit + '/quiz.html', '/api/unitquiz', '/favicon.ico') or p.startswith('/courses/shared/'):
            return False
        if p.startswith('/api/'):
            json_response(self, {'ok': False, 'error': 'Quiz in progress'}, 423)
            return True
        data = LOCKED_PAGE.replace('{unit}', unit).encode()
        self.send_response(423)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        return True

    def do_GET(self):
        try:
            p = urllib.parse.urlparse(self.path).path
            if self.blocked_by_quiz(p):
                return

            if p == '/':
                self.send_file(STATIC / 'student.html', 'text/html; charset=utf-8')
                return
            if p == '/teacher':
                if not self.is_teacher():
                    self.to_login()
                    return
                self.send_file(STATIC / 'teacher.html', 'text/html; charset=utf-8')
                return
            if p == '/login':
                self.send_file(STATIC / 'login.html', 'text/html; charset=utf-8')
                return
            if p == '/logout':
                self.redirect('/', 'teacher=; Path=/; Max-Age=0')
                return
            if p == '/api/whoami':
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                course = query.get('course', [''])[0]
                json_response(self, {'ok': True, 'teacher': self.is_teacher(course if course in course_names() else ALL)})
                return
            if p == '/app.js':
                self.send_file(STATIC / 'app.js', 'text/javascript; charset=utf-8')
                return

            # COURSE MATERIAL (static files under courses/)
            if p == '/courses' or p.startswith('/courses/'):
                target = course_path(p)
                if target is not None and target.is_dir():
                    if not p.endswith('/'):
                        self.send_response(301)
                        self.send_header('Location', p + '/')
                        self.end_headers()
                        return
                    target = target / 'index.html'
                if target is None or not target.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return
                if target.name in TEACHER_FILES and not self.is_teacher(scope_of(p)):
                    self.to_login()
                    return
                ctype = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
                if ctype.startswith('text/') or ctype.endswith(('json', 'javascript', 'svg+xml')):
                    ctype += '; charset=utf-8'
                self.send_file(target, ctype)
                return

            # UNIT QUIZ: questions without answers; options and question order shuffled per student
            if p == '/api/unitquiz':
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                quiz = load_unit_quiz(query.get('unit', [''])[0])
                if not quiz:
                    json_response(self, {'ok': False, 'error': 'Quiz not found'}, 404)
                    return
                public = []
                for item in quiz['questions']:
                    kind = item.get('type', 'mcq')
                    order = list(range(len(item['options'])))
                    if kind in ('mcq', 'multi') and not item.get('fixed'):
                        random.shuffle(order)
                    if kind == 'order':
                        while order == item['answer']:
                            random.shuffle(order)
                    public.append({
                        'id': item['id'], 'type': kind, 'topic': item.get('topic', ''),
                        'q': item['q'], 'code': item.get('code', ''), 'marks': item.get('marks', 1),
                        'options': [{'i': i, 't': item['options'][i]} for i in order],
                    })
                if quiz.get('shuffle_questions', True):
                    random.shuffle(public)
                json_response(self, {'ok': True, 'title': quiz['title'], 'minutes': quiz['minutes'],
                                     'instructions': quiz.get('instructions', []), 'questions': public})
                return

            if p == '/api/unitquiz/results':
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                unit = query.get('unit', [''])[0]
                if not self.is_teacher(scope_of('/courses/' + unit)):
                    json_response(self, {'ok': False, 'error': 'Teacher only'}, 403)
                    return
                quiz = load_unit_quiz(unit)
                if not quiz:
                    json_response(self, {'ok': False, 'error': 'Quiz not found'}, 404)
                    return
                with LOCK:
                    rows = read_results(unit)
                meta = [{'id': x['id'], 'topic': x.get('topic', ''), 'q': x['q'], 'marks': x.get('marks', 1)}
                        for x in quiz['questions']]
                json_response(self, {'ok': True, 'title': quiz['title'], 'questions': meta, 'results': rows})
                return
            
            if p == '/api/state':
                with LOCK:
                    topics = STATE['topics']
                    current_topic = topics[STATE['current_topic']] if topics else None
                    response_data = {
                        'topics': topics,
                        'current_topic': STATE['current_topic'],
                        'current_section': STATE['current_section'],
                        'current_topic_data': current_topic,
                        'students': list(STATE['students'].values()),
                        'student_count': len(STATE['students']),
                        'class_name': STATE['class_name'],
                        'teacher_name': STATE['teacher_name'],
                        'announcements': STATE['announcements']
                    }
                json_response(self, response_data)
                return
            
            if p == '/favicon.ico':
                self.send_response(204)
                self.end_headers()
                return
            
            self.send_response(404)
            self.end_headers()
        except:
            pass
    
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', '0'))
            if content_length > 0:
                data = json.loads(self.rfile.read(content_length) or '{}')
            else:
                data = {}
        except:
            json_response(self, {'ok': False, 'error': 'Invalid JSON'}, 400)
            return
        
        try:
            p = urllib.parse.urlparse(self.path).path
            print(f"POST {p}")

            # TEACHER LOGIN (for devices other than this laptop)
            if p == '/api/login':
                ip, now = self.client_address[0], time.time()
                with LOCK:
                    tries, first = FAILED.get(ip, (0, now))
                    if now - first > LOCKOUT:
                        tries, first = 0, now
                    if tries >= MAX_TRIES:
                        wait = int(LOCKOUT - (now - first)) // 60 + 1
                        json_response(self, {'ok': False, 'error': f'Too many wrong PINs. Try again in {wait} min.'}, 429)
                        return
                    pin, nxt = str(data.get('pin', '')).strip(), safe_next(str(data.get('next', '/teacher')))
                    scope = scope_of(nxt)
                    if secrets.compare_digest(pin, TEACHER_PIN):
                        granted = ALL
                    elif scope != ALL and secrets.compare_digest(pin, course_pin(scope)):
                        granted = scope
                    else:
                        FAILED[ip] = (tries + 1, first)
                        left = MAX_TRIES - tries - 1
                        hint = '' if scope == ALL else ' Use the PIN for this course or the master PIN.'
                        json_response(self, {'ok': False, 'error': f'Wrong PIN ({left} tries left).{hint}'}, 403)
                        return
                    FAILED.pop(ip, None)
                    token = self.session_token() or secrets.token_hex(16)   # one device can unlock several courses
                    SESSIONS.setdefault(token, set()).add(granted)
                data_out = json.dumps({'ok': True, 'next': nxt}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Set-Cookie', f'teacher={token}; Path=/; HttpOnly; SameSite=Strict')
                self.send_header('Content-Length', str(len(data_out)))
                self.end_headers()
                self.wfile.write(data_out)
                return

            if (p in TEACHER_POSTS or p.startswith('/api/teacher/')) and not self.is_teacher():
                json_response(self, {'ok': False, 'error': 'Teacher only'}, 403)
                return

            # STUDENT JOIN
            if p == '/api/join':
                name = str(data.get('name', '')).strip()
                if not name:
                    json_response(self, {'ok': False, 'error': 'Enter name'}, 400)
                    return
                sid = generate_id()
                with LOCK:
                    STATE['students'][sid] = {'id': sid, 'name': name}
                json_response(self, {'ok': True, 'id': sid})
                return
            
            # NAVIGATE
            if p == '/api/navigate':
                topic_index = int(data.get('topic', 0))
                section = data.get('section', 'content')
                with LOCK:
                    if STATE['topics']:
                        STATE['current_topic'] = max(0, min(topic_index, len(STATE['topics']) - 1))
                        STATE['current_section'] = section
                json_response(self, {'ok': True})
                return
            
            # ADD NEW TOPIC
            if p == '/api/teacher/add_topic':
                title = str(data.get('title', '')).strip()
                if not title:
                    json_response(self, {'ok': False, 'error': 'Title required'}, 400)
                    return
                
                new_topic = {
                    'id': generate_id(),
                    'title': title,
                    'description': data.get('description', ''),
                    'sections': {
                        'content': {
                            'title': title,
                            'body': ''  # Empty, teacher will add
                        },
                        'quiz': {
                            'question': '',
                            'options': ['', ''],
                            'correct': 0,
                            'explanation': '',
                            'points': 10,
                            'time_limit': 60
                        },
                        'assignment': {
                            'title': '',
                            'description': '',
                            'tasks': [],
                            'due_date': '',
                            'max_points': 100,
                            'instructions': ''
                        }
                    }
                }
                
                with LOCK:
                    STATE['topics'].append(new_topic)
                    STATE['current_topic'] = len(STATE['topics']) - 1
                
                json_response(self, {'ok': True, 'topic': new_topic})
                return
            
            # UPDATE CONTENT
            if p == '/api/teacher/update_content':
                topic_id = data.get('topic_id')
                content_data = data.get('content', {})
                with LOCK:
                    for topic in STATE['topics']:
                        if topic['id'] == topic_id:
                            topic['sections']['content'].update(content_data)
                            break
                json_response(self, {'ok': True})
                return
            
            # UPDATE QUIZ
            if p == '/api/teacher/update_quiz':
                topic_id = data.get('topic_id')
                quiz_data = data.get('quiz', {})
                with LOCK:
                    for topic in STATE['topics']:
                        if topic['id'] == topic_id:
                            topic['sections']['quiz'].update(quiz_data)
                            break
                json_response(self, {'ok': True})
                return
            
            # UPDATE ASSIGNMENT
            if p == '/api/teacher/update_assignment':
                topic_id = data.get('topic_id')
                assignment_data = data.get('assignment', {})
                with LOCK:
                    for topic in STATE['topics']:
                        if topic['id'] == topic_id:
                            topic['sections']['assignment'].update(assignment_data)
                            break
                json_response(self, {'ok': True})
                return
            
            # DELETE TOPIC
            if p == '/api/teacher/delete_topic':
                topic_id = data.get('topic_id')
                with LOCK:
                    STATE['topics'] = [t for t in STATE['topics'] if t['id'] != topic_id]
                    if STATE['current_topic'] >= len(STATE['topics']):
                        STATE['current_topic'] = max(0, len(STATE['topics']) - 1)
                json_response(self, {'ok': True})
                return
            
            # QUIZ SUBMIT (Student)
            if p == '/api/quiz/submit':
                sid = str(data.get('id', ''))
                answer = int(data.get('answer', -1))
                with LOCK:
                    if STATE['topics']:
                        topic = STATE['topics'][STATE['current_topic']]
                        quiz = topic['sections']['quiz']
                        correct = quiz.get('correct', 0)
                        is_correct = answer == correct
                        points = quiz.get('points', 10) if is_correct else 0
                        STATE['quizzes_results'][sid] = {'correct': is_correct, 'points': points}
                        json_response(self, {'ok': True, 'correct': is_correct, 'points': points})
                        return
                json_response(self, {'ok': False}, 400)
                return
            
            # ASSIGNMENT SUBMIT (Student)
            if p == '/api/assignment/submit':
                sid = str(data.get('id', ''))
                submission = str(data.get('submission', ''))
                with LOCK:
                    STATE['assignments'][sid] = {'submission': submission}
                json_response(self, {'ok': True})
                return
            
            # UNIT QUIZ START: lock this device to the quiz page until submit (see QUIZ_LOCKS)
            if p == '/api/unitquiz/start':
                unit = str(data.get('unit', ''))
                quiz = load_unit_quiz(unit)
                if not quiz:
                    json_response(self, {'ok': False, 'error': 'Quiz not found'}, 404)
                    return
                full = quiz['minutes'] * 60
                try:
                    left = min(full, max(0, int(data.get('seconds_left', full))))
                except (TypeError, ValueError):
                    left = full
                if not self.is_teacher():
                    with LOCK:
                        QUIZ_LOCKS[self.client_address[0]] = {'unit': unit, 'until': time.time() + left + QUIZ_GRACE}
                json_response(self, {'ok': True})
                return

            # UNIT QUIZ SUBMIT: graded here so the answer key never reaches the browser early
            if p == '/api/unitquiz/submit':
                unit = str(data.get('unit', ''))
                quiz = load_unit_quiz(unit)
                name = str(data.get('name', '')).strip()[:60]
                roll = str(data.get('roll', '')).strip()[:30]
                if not quiz or not name:
                    json_response(self, {'ok': False, 'error': 'Name and a valid quiz are required'}, 400)
                    return
                answers = data.get('answers') or {}
                review, score, total, per_q = [], 0.0, 0, {}
                for item in quiz['questions']:
                    marks = item.get('marks', 1)
                    frac = grade_question(item, answers.get(item['id']))
                    score += frac * marks
                    total += marks
                    per_q[item['id']] = round(frac, 2)
                    review.append({'id': item['id'], 'earned': round(frac * marks, 2), 'marks': marks,
                                   'answer': item['answer'], 'explain': item.get('explain', '')})
                try:
                    seconds = int(data.get('seconds') or 0)
                except (TypeError, ValueError):
                    seconds = 0
                record = {'name': name, 'roll': roll, 'score': round(score, 2), 'total': total,
                          'seconds': seconds, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'per_q': per_q}
                try:
                    record['left_page'] = max(0, int(data.get('left_page') or 0))
                    record['away_seconds'] = max(0, int(data.get('away_seconds') or 0))
                except (TypeError, ValueError):
                    record['left_page'] = record['away_seconds'] = 0
                with LOCK:
                    RESULTS.mkdir(exist_ok=True)
                    rows = read_results(unit)
                    rows.append(record)
                    results_file(unit).write_text(json.dumps(rows, indent=1), encoding='utf-8')
                    QUIZ_LOCKS.pop(self.client_address[0], None)
                json_response(self, {'ok': True, 'score': record['score'], 'total': total, 'review': review})
                return

            # ANNOUNCEMENT
            if p == '/api/teacher/announce':
                message = str(data.get('message', '')).strip()
                if message:
                    with LOCK:
                        STATE['announcements'].append({'message': message, 'time': time.time()})
                    json_response(self, {'ok': True})
                    return
                json_response(self, {'ok': False}, 400)
                return
            
            # RESET
            if p == '/api/reset':
                with LOCK:
                    STATE['current_topic'] = 0
                    STATE['current_section'] = 'content'
                    STATE['students'] = {}
                    STATE['assignments'] = {}
                    STATE['quizzes_results'] = {}
                json_response(self, {'ok': True})
                return
            
            json_response(self, {'ok': False, 'error': f'Unknown: {p}'}, 404)
        except Exception as e:
            print(f"Error: {e}")
            try:
                json_response(self, {'ok': False, 'error': str(e)}, 500)
            except:
                pass
    
    def log_message(self, *args):
        pass

if __name__ == '__main__':
    STATIC.mkdir(exist_ok=True)
    ip = ip_address()
    print(f'\n🎓 LIVE CLASSROOM')
    print(f'Students: http://{ip}:{PORT}')
    print(f'Teacher: http://{ip}:{PORT}/teacher')
    print(f'Courses: http://{ip}:{PORT}/courses/')
    print('\nPINs (only needed on devices other than this laptop):')
    print(f'  Master PIN, all courses + /teacher : {TEACHER_PIN}')
    for course in course_names():
        print(f'  {course:<34}: {course_pin(course)}   (change it in courses/{course}/_pin.txt)')
    print()
    
    class Reusable(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    
    with Reusable(('0.0.0.0', PORT), Handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('\nServer stopped')
            