import http.server, socketserver, json, socket, urllib.parse, threading
from pathlib import Path

PORT = 8000
ROOT = Path(__file__).parent
STATIC = ROOT / 'static'
LOCK = threading.Lock()
STATE = {'slide': 0, 'students': {}, 'poll': None, 'answers': {}}

LESSONS = [
('Welcome to C Control Flow', '''<h1>Welcome to C Control Flow</h1><p>Control flow means the order in which statements execute.</p><div class="concept">Today we learn how a C program decides what to execute.</div><h2>Learning path</h2><ol><li>Sequential flow</li><li>Independent if</li><li>if-else</li><li>else-if chain</li><li>Nested if</li><li>Compound conditions</li><li>switch-case</li><li>Loops and jumps</li></ol>'''),
('Sequential Flow', '''<h1>Sequential Flow</h1><p>Statements normally execute from top to bottom.</p><pre>int a = 10;\nint b = 20;\nint sum = a + b;\nprintf("%d", sum);</pre><div class="flow">Start → a = 10 → b = 20 → sum → printf → End</div><p><b>Key idea:</b> no decision and no repetition.</p>'''),
('Independent if', '''<h1>Independent if</h1><p>Each if is a separate decision. Both can execute.</p><pre>if(age >= 18)\n{\n    printf("Adult");\n}\n\nif(marks >= 50)\n{\n    printf("Passed");\n}</pre><div class="concept">Think: check the first decision, then independently check the second.</div>'''),
('Predict the Output', '''<h1>Predict the Output</h1><pre>int x = 10;\n\nif(x > 5)\n    printf("A");\n\nif(x > 8)\n    printf("B");\n\nif(x > 12)\n    printf("C");</pre><p>Use the teacher's poll to answer.</p>'''),
('Independent vs else-if', '''<h1>Independent vs else-if</h1><pre>if(x > 5)\n    printf("A");\n\nif(x > 8)\n    printf("B");</pre><p>Both decisions are checked.</p><hr><pre>if(x > 5)\n    printf("A");\nelse if(x > 8)\n    printf("B");</pre><p>Here the chain stops after the first true branch.</p>'''),
('Nested if', '''<h1>Nested if</h1><p>An if inside another if.</p><pre>if(age >= 18)\n{\n    if(hasID == 1)\n    {\n        printf("Entry allowed");\n    }\n}</pre><div class="flow">Age check → if true → ID check → if true → Entry allowed</div>'''),
('Quick Check', '''<h1>Quick Check</h1><p>Which structure means “check B only after reaching the A block”?</p><div class="option">A. Two independent if statements</div><div class="option">B. Nested if</div><div class="option">C. Two printf statements</div><div class="option">D. Sequential statements only</div>'''),
('Compound Conditions', '''<h1>Compound Conditions</h1><pre>if(age >= 18 && hasID == 1)\n{\n    printf("Allowed");\n}</pre><ul><li>&amp;&amp; means AND</li><li>|| means OR</li><li>! means NOT</li></ul>'''),
('switch-case', '''<h1>switch-case</h1><pre>switch(section)\n{\n    case 1:\n        printf("Section A");\n        break;\n    case 2:\n        printf("Section B");\n        break;\n    default:\n        printf("Invalid section");\n}</pre><p>Use switch for matching discrete values.</p>'''),
('Loops', '''<h1>Loops: Repetition</h1><p>Decisions answer: “Which path?”</p><p>Loops answer: “How many times?”</p><pre>for(int i = 1; i <= 5; i++)\n{\n    printf("%d\\n", i);\n}</pre>''')]

def ip_address():
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255',1)); return s.getsockname()[0]
    except Exception: return '127.0.0.1'
    finally: s.close()

def json_response(handler, obj, status=200):
    data=json.dumps(obj).encode()
    handler.send_response(status); handler.send_header('Content-Type','application/json; charset=utf-8'); handler.send_header('Cache-Control','no-store'); handler.send_header('Content-Length',str(len(data))); handler.end_headers(); handler.wfile.write(data)

class Handler(http.server.BaseHTTPRequestHandler):
    def send_file(self, path, content_type):
        data=path.read_bytes(); self.send_response(200); self.send_header('Content-Type',content_type); self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        p=urllib.parse.urlparse(self.path).path
        if p=='/': self.send_file(STATIC/'student.html','text/html; charset=utf-8'); return
        if p=='/teacher': self.send_file(STATIC/'teacher.html','text/html; charset=utf-8'); return
        if p=='/app.js': self.send_file(STATIC/'app.js','text/javascript; charset=utf-8'); return
        if p=='/style.css': self.send_file(STATIC/'style.css','text/css; charset=utf-8'); return
        if p=='/api/state':
            with LOCK:
                slide=STATE['slide']; poll=STATE['poll']; students=list(STATE['students'].values()); answers=dict(STATE['answers'])
            json_response(self, {'slide':slide,'total_slides':len(LESSONS),'lesson':{'title':LESSONS[slide][0],'content':LESSONS[slide][1]},'students':students,'student_count':len(students),'poll':poll,'answers':answers}); return
        if p=='/favicon.ico': self.send_response(204); self.end_headers(); return
        self.send_response(404); self.end_headers()
    def do_POST(self):
        try: data=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0')) or 0) or '{}')
        except Exception: json_response(self,{'ok':False,'error':'Invalid JSON'},400); return
        p=self.path
        if p=='/api/join':
            name=str(data.get('name','')).strip()
            if not name: json_response(self,{'ok':False,'error':'Please enter your name.'},400); return
            sid=f'{id(self)}-{name}'
            with LOCK: STATE['students'][sid]={'id':sid,'name':name}
            json_response(self,{'ok':True,'id':sid}); return
        if p=='/api/next':
            with LOCK: STATE['slide']=min(STATE['slide']+1,len(LESSONS)-1); STATE['poll']=None; STATE['answers']={}
            json_response(self,{'ok':True}); return
        if p=='/api/prev':
            with LOCK: STATE['slide']=max(STATE['slide']-1,0); STATE['poll']=None; STATE['answers']={}
            json_response(self,{'ok':True}); return
        if p=='/api/goto':
            n=int(data.get('slide',0));
            with LOCK: STATE['slide']=max(0,min(n,len(LESSONS)-1)); STATE['poll']=None; STATE['answers']={}
            json_response(self,{'ok':True}); return
        if p=='/api/poll':
            q=str(data.get('question','')).strip(); opts=data.get('options',[])
            if not q or not isinstance(opts,list) or len(opts)<2: json_response(self,{'ok':False,'error':'Question and at least two options required.'},400); return
            with LOCK: STATE['poll']={'question':q,'options':[str(x) for x in opts]}; STATE['answers']={}
            json_response(self,{'ok':True}); return
        if p=='/api/answer':
            sid=str(data.get('id','')); ans=str(data.get('answer',''))
            with LOCK:
                if sid in STATE['students'] and STATE['poll']: STATE['answers'][sid]=ans
            json_response(self,{'ok':True}); return
        if p=='/api/reset':
            with LOCK: STATE['slide']=0; STATE['students']={}; STATE['poll']=None; STATE['answers']={}
            json_response(self,{'ok':True}); return
        json_response(self,{'ok':False},404)
    def log_message(self,*args): pass

if __name__=='__main__':
    ip=ip_address(); print('\nOFFLINE CLASSROOM READY'); print(f'Students: http://{ip}:{PORT}'); print(f'Teacher:  http://{ip}:{PORT}/teacher'); print('Local test: http://127.0.0.1:8000'); print('Press Ctrl+C to stop.\n')
    class Reusable(socketserver.ThreadingTCPServer): allow_reuse_address=True
    with Reusable(('0.0.0.0',PORT),Handler) as server: server.serve_forever()
