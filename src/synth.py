"""Synthetic job ads (fake AND legitimate) in the EMSCAD column format, to add modern scam types the 2012-14 data lacks.

Fakes and legits come from the same generator on purpose: if only the fakes were synthetic the model would learn
"generator style = fake". Both draw locations, salary presence and polish from similar distributions, and the legits
include hard negatives (WhatsApp scheduling, M-Pesa pay, "urgent hiring", "no experience, we train").
Only ever used for TRAINING / extra checks. Real EMSCAD validation and test sets stay purely real.

    python src/synth.py        # prints samples and writes data/synthetic_train.csv + data/synthetic_holdout.csv
"""
import random
from pathlib import Path

import pandas as pd

COLS = ["title", "location", "company_profile", "description", "requirements", "benefits", "salary_range",
        "employment_type", "telecommuting", "has_company_logo", "has_questions", "required_experience",
        "required_education", "industry", "function", "fraudulent"]

LOCS = [("US", "NY", "New York"), ("US", "CA", "Los Angeles"), ("US", "TX", "Austin"), ("GB", "ENG", "London"), ("KE", "", "Nairobi"),
        ("KE", "", "Mombasa"), ("NG", "", "Lagos"), ("ZA", "", "Johannesburg"), ("IN", "KA", "Bangalore"), ("AE", "", "Dubai"),
        ("CA", "ON", "Toronto"), ("AU", "NSW", "Sydney"), ("PH", "", "Manila"), ("GH", "", "Accra"), ("UG", "", "Kampala"), ("DE", "", "Berlin")]
CUR = {"US": "$", "GB": "£", "KE": "KES ", "NG": "NGN ", "ZA": "ZAR ", "IN": "INR ", "AE": "AED ", "CA": "CAD ", "AU": "AUD ", "PH": "PHP ", "GH": "GHS ", "UG": "UGX ", "DE": "€"}
P1 = ["Apex", "Bluewave", "Cedar", "Delta", "Evergreen", "Falcon", "Granite", "Horizon", "Ivory", "Juniper", "Keystone", "Lumen", "Meridian", "Nova", "Orbit", "Pioneer", "Quantum", "Riverbend", "Summit", "Trident"]
P2 = ["Logistics", "Health", "Solutions", "Systems", "Foods", "Group", "Technologies", "Care", "Consulting", "Holdings", "Learning", "Motors", "Retail", "Finance", "Energy"]
PEOPLE = ["Grace", "Daniel", "Amina", "Peter", "Linda", "Samuel", "Joy", "Brian", "Mercy", "Kevin"]


def pick(r, xs, k=1):
    return r.choice(xs) if k == 1 else r.sample(xs, min(k, len(xs)))


# ---------------------------------------------------------------- legitimate roles: (title, industry, duties, requirements, edu, exp)
ROLES = [
    ("Registered Nurse", "Hospital & Health Care", ["Provide direct patient care on the ward", "Administer medication and record vitals", "Coordinate with doctors and families", "Keep accurate patient records"], ["Diploma or degree in nursing", "Valid registration with the nursing council", "2+ years of ward experience"], "Bachelor's Degree", "Mid-Senior level"),
    ("Accountant", "Accounting", ["Prepare monthly management accounts", "Reconcile bank and supplier statements", "Support the annual audit", "Process payroll and statutory returns"], ["Degree in accounting or finance", "CPA or ACCA qualification (in progress is fine)", "Experience with accounting software"], "Bachelor's Degree", "Associate"),
    ("Software Engineer", "Information Technology and Services", ["Build and maintain backend services", "Review code and write automated tests", "Work with product and design to ship features", "Monitor and improve system reliability"], ["Degree in computer science or equivalent experience", "Strong skills in Python, Java or JavaScript", "Familiarity with SQL and cloud platforms"], "Bachelor's Degree", "Mid-Senior level"),
    ("Sales Executive", "Retail", ["Meet agreed monthly sales targets", "Visit and onboard new retail customers", "Maintain relationships with existing accounts", "Report weekly on pipeline and market feedback"], ["Diploma or degree in business or marketing", "1+ year of field sales experience", "Valid driving licence is an advantage"], "Bachelor's Degree", "Entry level"),
    ("Delivery Rider", "Logistics and Supply Chain", ["Collect and deliver parcels within your zone", "Keep your delivery app and records up to date", "Handle cash-on-delivery payments honestly"], ["Own motorbike and valid licence", "Smartphone with data bundle", "Good knowledge of the city"], "High School or equivalent", "Entry level"),
    ("Primary School Teacher", "Education Management", ["Plan and teach lessons to Grades 3 to 5", "Assess pupils and report to parents", "Take part in school events and staff meetings"], ["Diploma or degree in education", "Registered with the teachers' commission", "Patience and strong classroom management"], "Bachelor's Degree", "Associate"),
    ("Warehouse Associate", "Logistics and Supply Chain", ["Receive, pick and pack customer orders", "Operate a pallet jack safely", "Count stock and report discrepancies", "Keep the warehouse clean and organised"], ["Ability to lift 20kg", "Basic literacy and numeracy", "Previous warehouse experience is an advantage"], "High School or equivalent", "Entry level"),
    ("Customer Service Representative", "Telecommunications", ["Answer customer calls and chats", "Resolve billing and service queries", "Log all contacts in the CRM", "Escalate complex issues to the right team"], ["Fluent English; Kiswahili or another local language is a plus", "Good typing speed", "Customer service experience preferred"], "High School or equivalent", "Entry level"),
    ("Mechanical Engineer", "Mechanical or Industrial Engineering", ["Design and test mechanical components", "Prepare drawings and specifications", "Support production teams on the factory floor", "Ensure compliance with safety standards"], ["Degree in mechanical engineering", "Experience with CAD software", "Registered or eligible for registration with the engineers' board"], "Bachelor's Degree", "Mid-Senior level"),
    ("Marketing Officer", "Marketing and Advertising", ["Plan and run social media campaigns", "Write copy for email and web", "Track campaign results and report monthly", "Coordinate events with vendors"], ["Degree in marketing or communications", "Portfolio of campaign work", "Experience with analytics tools"], "Bachelor's Degree", "Associate"),
    ("Data Entry Clerk", "Administrative", ["Enter customer records into our internal system", "Check entries for accuracy against source documents", "Prepare simple weekly reports", "Archive paper files"], ["Typing speed of at least 40 wpm", "Proficiency in Excel", "Attention to detail", "References from a previous employer"], "High School or equivalent", "Entry level"),
    ("Virtual Customer Support Agent", "Computer Software", ["Support customers by chat and email from home", "Follow our knowledge base and escalation guide", "Work a scheduled shift of 8 hours", "Take part in weekly team calls"], ["Reliable laptop and internet connection", "Excellent written English", "Interview and background check required before start"], "High School or equivalent", "Entry level"),
    ("Security Guard", "Security and Investigations", ["Patrol the premises and monitor CCTV", "Control access of visitors and vehicles", "Write incident reports", "Respond to alarms"], ["Certificate of good conduct", "Completed security training", "Physically fit"], "High School or equivalent", "Entry level"),
    ("Civil Engineer", "Civil Engineering", ["Supervise construction on site", "Prepare bills of quantities", "Ensure work meets design and safety standards", "Liaise with contractors and clients"], ["Degree in civil engineering", "Registered engineer or graduate engineer", "3+ years on building or road projects"], "Bachelor's Degree", "Mid-Senior level"),
    ("Graphic Designer", "Design", ["Create social media graphics and brochures", "Prepare print-ready files", "Work with the marketing team to keep our brand consistent"], ["Portfolio required", "Proficiency in Illustrator and Photoshop", "Eye for layout and colour"], "Bachelor's Degree", "Associate"),
    ("Pharmacist", "Pharmaceuticals", ["Dispense medicines and counsel patients", "Manage stock and expiry dates", "Comply with pharmacy regulations", "Supervise pharmaceutical technologists"], ["Degree in pharmacy", "Registered with the pharmacy board", "1+ year of retail or hospital experience"], "Bachelor's Degree", "Mid-Senior level"),
    ("Chef", "Restaurants", ["Prepare meals to the standard of our menu", "Manage kitchen stock and ordering", "Maintain food safety and hygiene", "Train junior kitchen staff"], ["Culinary diploma or equivalent experience", "Food handler's certificate", "Ability to work evenings and weekends"], "Certification", "Mid-Senior level"),
    ("Field Programme Officer", "Nonprofit Organization Management", ["Coordinate community training sessions", "Collect and report programme data", "Work with local partners and leaders", "Travel to field sites weekly"], ["Degree in development studies or a related field", "2+ years of NGO experience", "Fluency in English and a local language"], "Bachelor's Degree", "Associate"),
    ("Receptionist", "Administrative", ["Greet visitors and answer the phone", "Book meetings and manage the diary", "Handle incoming and outgoing mail", "Support the office manager"], ["Diploma in office administration", "Pleasant and professional manner", "Good Microsoft Office skills"], "Certification", "Entry level"),
    ("Data Analyst", "Information Technology and Services", ["Build dashboards and weekly reports", "Clean and analyse sales data", "Present findings to managers", "Write SQL queries"], ["Degree in statistics, maths or computing", "Strong SQL and Excel", "Experience with a BI tool"], "Bachelor's Degree", "Associate"),
    ("Content Writer", "Media Production", ["Write blog posts and product pages", "Edit and fact-check drafts", "Follow our style guide and deadlines"], ["Writing samples required", "Degree in journalism or English", "Understanding of SEO basics"], "Bachelor's Degree", "Entry level"),
]
LEGIT_PROFILE = ["{c} is a {size} company based in {city}. We have served customers for {yrs} years and employ about {n} people.",
                 "{c} provides {field} services across {region}. Our team of {n} professionals is committed to quality and integrity.",
                 "Founded in {yr}, {c} has grown from a small local business into a trusted name in {field}. We value fairness, safety and growth.",
                 "{c} is an equal opportunity employer. We are a registered company and operate from our offices in {city}."]
LEGIT_BEN = ["Medical cover", "Paid annual leave", "Pension contribution", "Training and development", "Staff discounts", "Transport allowance", "13th month bonus", "Flexible hours", "Performance bonus", "Free lunch"]
LEGIT_HOW = ["Apply through the careers page on our website.", "Please send your CV and cover letter to {mail}.", "Only shortlisted candidates will be contacted for an interview.",
             "Submit your application by the closing date. Shortlisted candidates will be interviewed in person or on video call.", "Applications are reviewed by our HR team. Please do not send original documents at this stage."]
LEGIT_HARD = ["We will contact shortlisted candidates on WhatsApp to schedule interviews.", "Salary is paid monthly to your bank account or M-Pesa.", "Urgent hiring: we need someone to start next month.",
              "No experience necessary; we provide full on-the-job training.", "Pay is weekly for this role.", "This is a remote position; you will need a laptop and stable internet.",
              "Send your CV to {gmail}.", "You will be paid per delivery plus a weekly bonus."]


def _company(r):
    return f"{pick(r, P1)} {pick(r, P2)}"


def _domain(c):
    return c.lower().replace(" ", "")


def _salary(r, cc, lo, hi):
    cur = CUR.get(cc, "$")
    return f"{lo}-{hi}"  # EMSCAD style: bare numbers


def legit(r):
    t, ind, duties, reqs, edu, exp = pick(r, ROLES)
    cc, st, city = pick(r, LOCS)
    c = _company(r)
    loc = ", ".join(x for x in (cc, st, city) if x)
    prof = pick(r, LEGIT_PROFILE).format(c=c, size=pick(r, ["growing", "family-owned", "leading", "regional", "established"]), city=city, yrs=r.randint(3, 40),
                                         n=r.choice([20, 45, 120, 300, 800]), field=ind.lower(), region=pick(r, ["East Africa", "the region", "the country", "several cities"]), yr=r.randint(1985, 2018))
    mail = f"careers@{_domain(c)}.com"
    opener = pick(r, [f"{c} is looking for a {t} to join our team in {city}. ", f"Vacancy: {t}, {city}. ", f"We are hiring a {t}. ", f"{c} is recruiting a {t} for our {city} office. ", ""])
    desc = opener + ". ".join(pick(r, duties, r.randint(2, 4))) + ". "
    if r.random() < 0.35:
        desc += pick(r, LEGIT_HARD).format(gmail=f"{_domain(c)}hr@gmail.com") + " "
    desc += pick(r, LEGIT_HOW).format(mail=mail)
    if r.random() < 0.1:
        desc = desc.replace("Apply through", "Apply today through").upper() if r.random() < 0.15 else desc + " Hurry, applications close soon!"
    lo = r.choice([20, 30, 45, 60, 90]) * 1000
    return dict(title=t, location=loc, company_profile=prof if r.random() < 0.8 else None, description=desc,
                requirements="; ".join(reqs) if r.random() < 0.9 else None,
                benefits=", ".join(pick(r, LEGIT_BEN, r.randint(2, 4))) if r.random() < 0.6 else None,
                salary_range=_salary(r, cc, lo, lo + r.choice([10, 20, 40]) * 1000) if r.random() < 0.3 else None,
                employment_type=pick(r, ["Full-time", "Full-time", "Part-time", "Contract", None]), telecommuting=int(r.random() < 0.08),
                has_company_logo=int(r.random() < 0.65), has_questions=int(r.random() < 0.4), required_experience=exp, required_education=edu,
                industry=ind, function=None, fraudulent=0)


# ---------------------------------------------------------------- scams: type -> (titles, hooks, task lines, how-it-works lines)
PAY = ["{c}{a} per week", "{c}{a} daily", "{c}{a} per hour", "{c}{a} after your first week", "up to {c}{a} every month"]
URGENT = ["Limited slots available, apply today!", "We are hiring immediately, start this week.", "Only 10 positions left!", "Apply within 24 hours to secure your place.", "Urgent: first come, first served."]
CONTACT = ["WhatsApp {ph} to start", "Send a message to our recruiter on Telegram {tg}", "Email your CV to {gm}", "Text {ph} with the word JOB", "Contact our HR officer {name} on WhatsApp {ph}", "Click the link to register: {url}"]
SCAMS = {
    "fake_check": (["Remote Administrative Assistant", "Home Office Assistant", "Remote Customer Support Associate", "Part-Time Remote Data Coordinator"],
                   ["We are expanding and urgently need remote assistants.", "Join our remote team and work comfortably from home."],
                   ["After you are hired we will mail you a cheque to purchase your home-office equipment and software from our approved vendor.", "Our finance team will send you a cheque of {c}{eq} to buy your laptop from our preferred supplier. Deposit it, then send the balance to our vendor.",
                    "You will receive an advance payment to set up your workstation. Keep your pay and transfer the rest to the equipment vendor."]),
    "advance_fee": (["Office Assistant", "Cleaner - Overseas Hotel", "Hotel Receptionist", "Factory Worker", "Care Assistant Abroad"],
                    ["Guaranteed job placement. No experience needed.", "Our agency places candidates in well paying jobs quickly."],
                    ["A one-time registration fee of {c}{fee} is required to process your application.", "Pay a refundable {c}{fee} training and uniform fee via mobile money to confirm your slot.",
                     "Candidates must pay a {c}{fee} medical and processing fee before the interview. Your fee is refunded after the first salary."]),
    "phishing_interview": (["Remote Marketing Assistant", "Junior Project Coordinator", "Online Recruiter", "Customer Care Agent"],
                           ["Your profile was selected from a job board.", "We reviewed your CV and you are shortlisted."],
                           ["The interview will be held entirely on Telegram. Do not use video call. Our hiring manager will message you.", "Complete this online assessment on the link below, then send a photo of your ID, your bank details and your SSN for onboarding.",
                            "To finish onboarding, send a copy of your national ID, passport and bank account number so we can set up your payroll."]),
    "task_scam": (["Online Product Promoter", "Social Media Booster", "App Rating Assistant", "E-commerce Task Associate"],
                  ["Earn money by completing simple tasks on your phone.", "Like, rate and boost products to earn commission."],
                  ["Complete 40 tasks a day and earn commission. To unlock higher-paying tasks you top up your account with a small deposit that is refunded with profit.", "Earn {c}{a} daily just by rating products. Deposit {c}{fee} to activate your task account and withdraw anytime.",
                   "Our team lead will add you to a Telegram group. Recharge your account to continue earning bigger commissions."]),
    "reshipping": (["Package Handler - Work From Home", "Shipping Coordinator", "Parcel Inspector (Home Based)", "Logistics Assistant - Remote"],
                   ["Receive packages at your home and forward them for us.", "No experience required, flexible hours."],
                   ["You will receive parcels at your address, repackage them and ship them to our overseas clients. We pay for shipping.", "Packages are delivered to you, you inspect them and reship using our prepaid labels. Paid monthly after 30 days."]),
    "mystery_shopper": (["Mystery Shopper", "Secret Shopper - Paid Weekly", "Retail Evaluator"],
                        ["Get paid to shop and evaluate stores.", "Keep what you buy plus earn {c}{a} per assignment."],
                        ["We send you a cheque to cover your purchase. Cash it, shop, and wire the remaining funds through a money transfer service.", "You will be sent funds in advance. Evaluate the money transfer counter by sending a test payment."]),
    "data_entry_earn": (["Data Entry Clerk - Work From Home", "Online Typist", "Form Filling Job", "Copy Paste Jobs"],
                        ["Earn money from home. No experience, no skills, no interview.", "Make easy money online in your free time."],
                        ["Simply type and submit forms. Payments are made weekly. A small activation fee is needed for your account and software licence.", "Work from your phone, anytime, with unlimited earning potential. Registration is quick."]),
    "ceo_assistant": (["Personal Assistant to the CEO", "Executive Errand Assistant", "Confidential Assistant"],
                      ["Our CEO is travelling and needs a trustworthy assistant right away.", "Discreet help needed for urgent personal errands."],
                      ["You will buy gift cards for clients and send us the codes. We reimburse you with a generous bonus.", "Tasks include purchasing prepaid cards and sharing the numbers by message. Do not discuss this with others."]),
    "crypto_trainee": (["Forex Trading Trainee", "Crypto Account Assistant", "Investment Support Agent"],
                       ["Learn to trade and earn passive income.", "Guaranteed returns with our expert team."],
                       ["Start by funding your trading account with {c}{fee}. Our expert will trade for you and you keep 70% of the profits, guaranteed.", "Earn {c}{a} per day with zero risk. First deposit your starter capital in bitcoin to our wallet."]),
    "free_trial_work": (["Freelance Writer - Paid Trial", "Graphic Designer (Test Project)", "Video Editor Needed", "Social Media Manager - Trial Month"],
                        ["Great exposure and long-term opportunity.", "Complete a sample project to qualify for the position."],
                        ["Candidates complete a full unpaid test project of 3 to 5 days. Selected work is used for our clients. Successful applicants will be contacted later.", "Submit 5 original designs for our live campaigns as your trial. Payment will be discussed after the trial month."]),
}
GENERIC_PROFILE = ["We are a fast-growing international company with offices worldwide and thousands of satisfied employees.", "{c} is a leading global firm offering flexible work to people everywhere.",
                   "Our company has been operating for over 10 years and helps people earn from home.", "{c} is a registered company dedicated to creating job opportunities."]


def scam(r):
    kind = r.choice(list(SCAMS))
    titles, hooks, bits = SCAMS[kind]
    cc, st, city = pick(r, LOCS)
    c = _company(r)
    cur = CUR.get(cc, "$")
    scale = {"KES ": 10, "NGN ": 100, "INR ": 50, "UGX ": 300, "PHP ": 30, "ZAR ": 5, "GHS ": 5}.get(cur, 1)
    fmt = dict(c=cur, a=f"{r.choice([50, 100, 150, 200, 300, 500]) * scale:,}", fee=f"{r.choice([10, 20, 30, 50, 100]) * scale:,}", eq=f"{r.choice([2000, 2800, 3500]) * scale:,}")
    ph = f"+{r.choice([254, 234, 1, 44, 27, 91])} {r.randint(700, 799)} {r.randint(100, 999)} {r.randint(100, 999)}"
    ctx = dict(ph=ph, tg="@" + _domain(c) + "_hr", gm=f"{pick(r, PEOPLE).lower()}.{_domain(c)}@gmail.com", url=f"http://{_domain(c)}-jobs.com/apply", name=pick(r, PEOPLE), **fmt)
    parts = [pick(r, hooks).format(**fmt), pick(r, bits).format(**fmt)]
    if r.random() < 0.35:  # same openers as the legit ads, so the opener isn't a giveaway
        parts.insert(0, pick(r, [f"{c} is looking for a {titles[0]} to join our team in {city}.", f"Vacancy: {titles[0]}, {city}.", f"We are hiring a {titles[0]}.", f"{c} is recruiting a {titles[0]} for our {city} office."]))
    if r.random() < 0.3:
        parts.append(pick(r, ["Shortlisted candidates will be contacted.", "Applications are reviewed by our HR team.", "Please do not send original documents at this stage.", "Submit your application by the closing date."]))
    if r.random() < 0.6:
        parts.append("Pay: " + pick(r, PAY).format(**fmt) + ".")
    if r.random() < 0.55:
        parts.append(pick(r, URGENT))
    if r.random() < 0.5:
        parts.append(pick(r, ["No experience required.", "No interview needed.", "Anyone can apply.", "Training is provided, you start earning immediately."]))
    parts.append(pick(r, CONTACT).format(**ctx) + ".")
    desc = " ".join(parts)
    if r.random() < 0.25:
        desc = desc.upper() if r.random() < 0.3 else desc.replace(".", "!")
    t = pick(r, titles)
    polished = r.random() < 0.55  # many modern scams look professional
    return dict(title=t + (" !!!" if r.random() < 0.08 else ""), location=", ".join(x for x in (cc, st, city) if x) if r.random() < 0.85 else "Remote",
                company_profile=pick(r, GENERIC_PROFILE).format(c=c) if r.random() < (0.6 if polished else 0.35) else None, description=desc,
                requirements=pick(r, ["Smartphone or laptop; good internet", "Must be 18+; honest and hardworking", "Basic English", "Bank account or mobile money wallet"]) if r.random() < (0.7 if polished else 0.45) else None,
                benefits=pick(r, ["Weekly payments", "Flexible hours; work from anywhere", "Bonuses and commissions", "Free training"]) if r.random() < 0.5 else None,
                salary_range=f"{r.choice([2, 3, 5, 8]) * 1000 * scale}-{r.choice([10, 15, 20]) * 1000 * scale}" if r.random() < 0.3 else None,
                employment_type=pick(r, ["Part-time", "Full-time", "Contract", "Other", None]), telecommuting=int(r.random() < 0.5),
                has_company_logo=int(r.random() < (0.55 if polished else 0.2)), has_questions=int(r.random() < 0.25),
                required_experience=pick(r, ["Entry level", "Not Applicable", None]), required_education=pick(r, ["High School or equivalent", None]),
                industry=pick(r, ["Marketing and Advertising", "Administrative", "Financial Services", "Retail", None]), function=None, fraudulent=1)


def make(n_fake, n_legit, seed):
    r = random.Random(seed)
    rows = [scam(r) for _ in range(n_fake)] + [legit(r) for _ in range(n_legit)]
    r.shuffle(rows)
    df = pd.DataFrame(rows, columns=COLS)
    df.insert(0, "job_id", range(len(df)))
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "data"
    tr, ho = make(1200, 1200, seed=1), make(300, 300, seed=2)
    ho = ho[~ho.description.isin(set(tr.description))]  # no exact copies between train and holdout
    tr.to_csv(root / "synthetic_train.csv", index=False)
    ho.to_csv(root / "synthetic_holdout.csv", index=False)
    print(len(tr), len(ho), tr.fraudulent.mean())
    for lab in (1, 0):
        for _, x in tr[tr.fraudulent == lab].sample(3, random_state=7).iterrows():
            print(f"\n[{'FAKE' if lab else 'REAL'}] {x.title} | {x.location}\n{x.description}")
    assert tr.description.nunique() > 1500, tr.description.nunique()
    print("\nself-check ok; distinct descriptions:", tr.description.nunique())
