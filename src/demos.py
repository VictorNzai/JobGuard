"""Form defaults and rotating demo ads, shared by the Home and Check pages."""
import random

BLANK = dict(url="", title="", location="", link="", company_profile="", description="", requirements="", benefits="",
             salary="", employment_type="", logo=False, questions=False, remote=False)

FAKE = [
    dict(title="Data Entry Clerk - Work From Home", location="KE, Nairobi", link="http://bit.ly/earn-fast-apply", remote=True,
         description="Earn $500 weekly from home! No experience needed. Send your CV and ID copy to hr.jobs@gmail.com today!"),
    dict(title="Office Assistant (Urgent)", location="KE, Mombasa",
         description="We are hiring 20 office assistants immediately. Pay a refundable registration fee of KES 1,500 via M-Pesa Paybill 247247 "
                     "to secure your slot. Send a copy of your national ID and KRA PIN. Apply now, limited slots only!"),
    dict(title="Overseas Jobs: Dubai Hotel Staff", location="KE, Nairobi", employment_type="Full-time",
         description="Guaranteed jobs in Dubai and Qatar! Visa and air ticket processing fee KES 25,000 only. Salary 1,800 AED monthly. "
                     "No experience required. WhatsApp +254 700 123 456 to book your placement. First come, first served."),
    dict(title="Online Chat Support Agent", location="KE, Remote", remote=True, link="https://t.ly/chat-jobs",
         description="Make KES 5,000 per day replying to messages from your phone. No qualifications needed, anyone can do it. "
                     "Training kit costs KES 2,000 payable by M-Pesa. Contact us on Telegram to start earning today."),
    dict(title="Forex Trading Assistant", location="US, NY, New York", remote=True, salary="$3000-$8000",
         description="Unlimited earning potential! Our traders make guaranteed profits daily. Deposit $200 to open your trading account "
                     "and earn passive income from home. Financial freedom starts here. Email apply.forex.team@yahoo.com"),
    dict(title="Mystery Shopper - Get Paid To Shop!!!", location="US, TX, Dallas", salary="$300 per assignment",
         description="Get paid to shop and keep the items! We send you a cheque for $2,000. Deposit it, keep $300 as your pay and wire the "
                     "rest by Western Union to our evaluator. No experience needed. Reply with your full name, address and bank details."),
    dict(title="Typist Needed - Earn Daily", location="KE, Kisumu", remote=True,
         description="Typing jobs available. Earn KES 3,000 daily doing simple typing from home. No skills required. Send your ID copy and "
                     "pay KES 500 account activation fee to Till Number 567890. Hurry, apply today!"),
    dict(title="Personal Assistant to CEO", location="GB, LND, London", salary="£4,000 per week",
         description="Wealthy CEO seeks a personal assistant, work from home, no experience. Weekly pay £4,000. Send your CV, passport copy "
                     "and bank account details to ceo.office.recruit@gmail.com for immediate processing. Act now!"),
]
REAL = [
    dict(title="Senior Software Engineer", location="US, CA, San Francisco", employment_type="Full-time", logo=True, questions=True,
         link="https://www.examplecorp.com/careers/senior-software-engineer",
         company_profile="We are a growing software company with 200 employees building payments tools for clients worldwide.",
         description="Design and build backend services in Python. Work with product and design teams to ship features. Participate in code reviews and mentor junior engineers.",
         requirements="5+ years of experience with Python, SQL and cloud platforms. Bachelor's degree in computer science or equivalent.",
         benefits="Health insurance, 401k, flexible hours."),
    dict(title="Registered Nurse - Medical Ward", location="KE, Nairobi", employment_type="Full-time", logo=True, questions=True,
         company_profile="Mater Hospital is a leading private hospital in Nairobi, providing quality healthcare to patients since 1962.",
         description="Provide nursing care to patients on the medical ward, administer medication, keep accurate patient records and work "
                     "closely with doctors and the wider care team.",
         requirements="Diploma or degree in nursing, valid KRCHN registration and licence from the Nursing Council of Kenya, 2 years of ward experience.",
         benefits="Medical cover, pension scheme, annual leave and continuing education support."),
    dict(title="Accountant", location="GB, LND, London", employment_type="Full-time", logo=True, questions=True,
         link="https://www.brightfieldgroup.co.uk/careers",
         company_profile="Brightfield Group is a family-owned property management firm managing more than 3,000 homes across London.",
         description="Own the monthly close for our property portfolio, prepare management accounts, reconcile bank and ledger balances, "
                     "and support the annual audit.",
         requirements="ACA or ACCA qualified, at least 3 years in an accounting role, strong Excel skills.",
         benefits="25 days holiday, private health, cycle to work scheme.", salary="45000-55000"),
    dict(title="Primary School Teacher (Mathematics)", location="KE, Nakuru", employment_type="Full-time", logo=True,
         company_profile="Hillcrest Academy is a co-educational primary school in Nakuru with 600 pupils and a record of strong KCPE results.",
         description="Teach mathematics to Class 5 to 8, prepare lesson plans and assessments, mark and report on pupil progress, "
                     "and take part in school activities.",
         requirements="Diploma or degree in Education, registration with the Teachers Service Commission, at least 2 years of teaching experience.",
         benefits="Housing allowance, medical cover, professional development."),
    dict(title="Marketing Coordinator", location="US, NY, New York", employment_type="Full-time", logo=True, questions=True,
         link="https://www.northwind-foods.com/jobs/marketing-coordinator",
         company_profile="Northwind Foods is a regional food distributor with 450 employees serving restaurants and grocers across the Northeast.",
         description="Plan and run email and social campaigns, coordinate trade show logistics, maintain the product catalogue and "
                     "report on campaign results each month.",
         requirements="Bachelor's degree in marketing or communications, 2 years of experience, comfort with analytics tools.",
         benefits="Health and dental insurance, 401k match, 20 days paid time off."),
    dict(title="Logistics Officer", location="KE, Mombasa", employment_type="Full-time", logo=True, questions=True,
         company_profile="Coastline Freight handles import and export clearing at the Port of Mombasa for over 150 corporate clients.",
         description="Coordinate container clearing and delivery schedules, liaise with KRA and port authorities, track shipments in our "
                     "system and resolve delays with clients and transporters.",
         requirements="Diploma in supply chain or logistics, 3 years of clearing and forwarding experience, knowledge of customs procedures.",
         benefits="Medical insurance, NSSF and pension contributions, transport allowance."),
    dict(title="Customer Support Specialist", location="IE, D, Dublin", employment_type="Full-time", logo=True, questions=True, remote=True,
         link="https://www.cloudlane.io/careers/support",
         company_profile="Cloudlane provides project management software to 12,000 teams and has offices in Dublin and Austin.",
         description="Answer customer questions by email and chat, troubleshoot account issues, document common problems and pass product "
                     "feedback to our engineering team.",
         requirements="1 to 3 years in a customer-facing support role, clear written English, experience with a helpdesk tool.",
         benefits="Hybrid working, learning budget, private health insurance."),
    dict(title="Junior Data Analyst", location="US, MA, Boston", employment_type="Full-time", logo=True, questions=True,
         link="https://www.harborhealth.org/careers/data-analyst",
         company_profile="Harbor Health is a nonprofit health system with 12 clinics, focused on improving outcomes in underserved communities.",
         description="Build weekly dashboards, clean and validate clinical data, answer ad hoc questions from care teams and "
                     "document data definitions.",
         requirements="Bachelor's degree in statistics, public health or a related field, SQL and Excel, interest in healthcare.",
         benefits="Medical, dental and vision, tuition support, retirement plan."),
]
ADS = {"fake": FAKE, "real": REAL}


def pick(kind, state):
    """A random demo ad of this kind, never the same one twice in a row. `state` is st.session_state."""
    last = state.get(f"_last_{kind}")
    i = random.choice([n for n in range(len(ADS[kind])) if n != last])
    state[f"_last_{kind}"] = i
    return dict(BLANK, **ADS[kind][i])
