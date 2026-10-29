OFFLINE CLASSROOM - CLEAN VERSION

1. Open this folder in VS Code.
2. Terminal -> New Terminal.
3. Run: python server.py
4. Give students the "Students" / "Courses" address shown in the terminal.
5. Open /teacher on your laptop for teacher control.

LOCAL TEST:
http://127.0.0.1:8000            live classroom (students)
http://127.0.0.1:8000/teacher    teacher panel
http://127.0.0.1:8000/courses/   course library (study material, slides, quizzes)

LAB:
Connect the laptop by Ethernet/Wi-Fi to the same LAN as student PCs.
Give students the LAN address printed by server.py.
No internet is required.

TEACHER-ONLY PAGES
The teacher panel (/teacher) and every unit's board.html, session.html and results.html
open only on the laptop running server.py. Students get a PIN screen instead.
On any other device (e.g. a smart board with its own computer) enter a PIN:
  - Course PIN: each course has its own, in courses/<course>/_pin.txt (made on first start).
    It opens only that course's slides, timer and results. Give it to that course's teacher.
    To change it, edit the file and restart the server.
  - Master PIN: opens every course and /teacher. A new one is made at each start; to keep
    one fixed run  set TEACHER_PIN=482913  (PowerShell: $env:TEACHER_PIN="482913")  first.
All PINs are printed in the terminal when the server starts. _pin.txt files are never served
to browsers and are left out of git. 5 wrong PINs lock that device out for 10 minutes.
Open /logout to sign a device out.

COURSE MATERIAL (courses/ folder)
courses/
  index.html                         list of all courses
  shared/                            theme, scripts and diagram helpers used by every unit
  bca5-data-science/
    index.html                       list of units
    unit-1/
      index.html                     unit home with the 2-hour plan
      board.html                     smart-board slides (arrows, D = draw, N = notes, F = full screen)
      study.html                     35-minute study guide + in-browser lab
      quiz.html                      student quiz (graded on the server)
      results.html                   live quiz results for the teacher (CSV export)
      session.html                   full-screen timer for the whole session
      _quiz.json                     question bank WITH answers (files starting with _ are never served)
      lab/                           dataset, notebook and Python script for students
      img/                           charts used by the study guide
    unit-2/                          Statistics for Data Science: same pages; lab/ has the 600-order delivery
                                     trial + atta packet weights (Z, T, Chi-square, ANOVA in SciPy)
    Rebuild data and charts (deterministic): python tools/build_unit1_assets.py / build_unit2_assets.py
  bca1-ai-ml/                        Problem Solving Techniques Using C (BCA AI & ML, Sem 1)
    unit-1/                          same pages as above, plus worksheet.html (printable)
      lab/                           sum_steps.c, search_steps.c, grade_check.c
  bca3-ds-ai/                        Predictive Analytics Modeler (BCA DS & AI with IBM, Sem 3), all 5 units
    unit-1 … unit-5/                 same pages as above; each lab/ has data, Python script + notebook, R script
      unit-1  Introduction to Predictive Analytics   (SwiftTel churn data)
      unit-2  Data Preparation & Exploration          (3 messy SwiftTel source files)
      unit-3  Predictive Modeling Techniques          (used-car prices + SwiftTel churn)
      unit-4  Advanced Modeling & Validation          (SwiftTel churn; JSON results for the labs)
      unit-5  Model Deployment & Applications         (credit, clinic, segmentation; saved model;
                                                       lab/score_api.py = real-time scoring demo)
    Rebuild data and charts (deterministic): python tools/build_bca3_unitN_assets.py  (N = 1 … 5, in order)
    The R scripts were not run on the teacher laptop (R not installed): test them once in RStudio.

Course and unit folder names must use only lowercase letters, numbers and dashes
(no underscores), or the quiz will not load.

Quiz submissions are saved in quiz_results/<course>__<unit>.json.

ADDING THE NEXT UNIT
1. Copy unit-1 to unit-2 (folder name: lowercase letters, numbers, dashes).
2. Replace the content of each page and _quiz.json.
3. In quiz.html, results.html and session.html change UNIT / KEY to 'bca5-data-science/unit-2'.
4. In bca5-data-science/index.html turn the Unit 2 card into a link (copy the Unit 1 card).
