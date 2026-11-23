"""Fetch a job-posting URL and pull out title/company/location/text. Stdlib only."""
import ipaddress, json, socket, ssl, urllib.request
from html.parser import HTMLParser
from urllib.parse import urlparse

SKIP = {"script", "style", "noscript", "nav", "header", "footer", "svg", "form"}
MAX_BYTES = 2_000_000
try:  # certifi's CA list avoids "certificate verify failed" on some Windows Pythons
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context()


class _P(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip, self.text, self.title, self.og_title = 0, [], "", ""
        self.ld, self._in_ld, self._in_title, self.has_image = [], False, False, False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta":
            if a.get("property") == "og:title":
                self.og_title = a.get("content") or ""
            if a.get("property") == "og:image":
                self.has_image = True
        if tag == "script" and a.get("type") == "application/ld+json":
            self._in_ld = True
        elif tag in SKIP:
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag in ("p", "li", "br", "div", "h1", "h2", "h3", "tr"):
            self.text.append("\n")

    def handle_endtag(self, tag):
        if tag == "script" and self._in_ld:
            self._in_ld = False
        elif tag in SKIP and self.skip:
            self.skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, d):
        if self._in_ld:
            self.ld.append(d)
        elif self._in_title:
            self.title += d
        elif not self.skip:
            self.text.append(d)


def _jobposting(ld_blocks):
    """First schema.org JobPosting found in the JSON-LD blocks, else None."""
    def walk(x):
        if isinstance(x, list):
            for i in x:
                yield from walk(i)
        elif isinstance(x, dict):
            t = x.get("@type")
            if t == "JobPosting" or (isinstance(t, list) and "JobPosting" in t):
                yield x
            for v in x.values():
                yield from walk(v)
    for raw in ld_blocks:
        try:
            for j in walk(json.loads(raw)):
                return j
        except ValueError:
            pass


def _strip(h):
    p = _P()
    p.feed(h)
    return " ".join("".join(p.text).split())


def parse_job_html(html: str) -> dict:
    p = _P()
    p.feed(html)
    body = "\n".join(" ".join(l.split()) for l in "".join(p.text).splitlines() if l.strip())
    out = dict(title=(p.og_title or p.title).strip(), company="", location="", description=body,
               employment_type="", salary="", has_image=p.has_image, structured=False)
    j = _jobposting(["".join(p.ld)] if p.ld else [])
    if j:
        out["structured"] = True
        out["title"] = j.get("title") or out["title"]
        if j.get("description"):
            out["description"] = _strip(j["description"])
        org = j.get("hiringOrganization")
        out["company"] = (org.get("name") if isinstance(org, dict) else org) or ""
        loc = j.get("jobLocation")
        loc = loc[0] if isinstance(loc, list) and loc else loc
        addr = loc.get("address") if isinstance(loc, dict) else None
        if isinstance(addr, dict):
            out["location"] = ", ".join(str(addr[k]) for k in ("addressCountry", "addressRegion", "addressLocality") if addr.get(k))
        et = j.get("employmentType")
        et = et[0] if isinstance(et, list) and et else et
        out["employment_type"] = {"FULL_TIME": "Full-time", "PART_TIME": "Part-time", "CONTRACTOR": "Contract",
                                  "TEMPORARY": "Temporary"}.get(str(et).upper(), "")
        sal = j.get("baseSalary")
        out["salary"] = "yes" if sal else ""
    out["description"] = out["description"][:8000]
    return out


def _check_public(host):
    for info in socket.getaddrinfo(host, None):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("That address is on a private network; only public job pages can be fetched.")


def fetch_job(url: str) -> dict:
    u = urlparse(url if "://" in url else "https://" + url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("Enter a full http(s) link.")
    _check_public(u.hostname)
    req = urllib.request.Request(u.geturl(), headers={"User-Agent": "Mozilla/5.0 (JobGuard student project)"})
    with urllib.request.urlopen(req, timeout=12, context=CTX) as r:
        raw = r.read(MAX_BYTES)
        cs = r.headers.get_content_charset() or "utf-8"
    page = parse_job_html(raw.decode(cs, errors="replace"))
    if len(page["description"]) < 80:
        raise ValueError("Couldn't read any ad text from that page (it may need a login or load its content with JavaScript). Paste the text instead.")
    return page


if __name__ == "__main__":
    h = ('<html><head><title>x</title><script type="application/ld+json">{"@type":"JobPosting","title":"Nurse",'
         '"description":"<p>Care for patients</p>","hiringOrganization":{"name":"ACME"},"employmentType":"FULL_TIME",'
         '"jobLocation":{"address":{"addressCountry":"KE","addressLocality":"Nairobi"}}}</script></head>'
         '<body><nav>menu</nav><p>Hello</p></body></html>')
    r = parse_job_html(h)
    assert r["title"] == "Nurse" and r["company"] == "ACME" and r["location"] == "KE, Nairobi"
    assert r["description"] == "Care for patients" and r["employment_type"] == "Full-time"
    r = parse_job_html("<title>T</title><nav>menu</nav><script>var a</script><p>Real text</p>")
    assert r["title"] == "T" and r["description"] == "Real text", r
    try:
        fetch_job("http://127.0.0.1:8501")
        raise SystemExit("private IP not blocked")
    except ValueError:
        pass
    print("ok")
