import http.server, socketserver, json, socket, urllib.parse, threading
from pathlib import Path
import uuid
import time

PORT = 8000
ROOT = Path(__file__).parent
STATIC = ROOT / 'static'
LOCK = threading.Lock()

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
    
    def do_GET(self):
        try:
            p = urllib.parse.urlparse(self.path).path
            
            if p == '/':
                self.send_file(STATIC / 'student.html', 'text/html; charset=utf-8')
                return
            if p == '/teacher':
                self.send_file(STATIC / 'teacher.html', 'text/html; charset=utf-8')
                return
            if p == '/app.js':
                self.send_file(STATIC / 'app.js', 'text/javascript; charset=utf-8')
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
    print(f'Teacher: http://{ip}:{PORT}/teacher\n')
    
    class Reusable(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    
    with Reusable(('0.0.0.0', PORT), Handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('\nServer stopped')
            