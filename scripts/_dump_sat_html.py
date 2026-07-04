import re
import urllib.request

qid = "886dc9f9"
url = (
    "https://sat-questions.onrender.com/question/"
    f"module:english-group:all-skill:all-difficulty:all-active:all/{qid}"
)
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
open(r"C:\Users\Fayzulloh\adaptivprep\scripts\_sample_sat.html", "w", encoding="utf-8").write(html)
for m in re.finditer(r".{0,40}Correct.{0,80}", html):
    print(m.group(0)[:120])
