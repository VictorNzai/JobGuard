"""Form defaults and the two demo ads, shared by the Home and Check pages."""
BLANK = dict(title="", location="", link="", company_profile="", ad="", requirements="", benefits="",
             salary="", employment_type="", logo=False, questions=False, remote=False)
DEMOS = {
    "fake": dict(BLANK, title="Data Entry Clerk - Work From Home", location="KE, Nairobi",
                 link="http://bit.ly/earn-fast-apply", remote=True,
                 ad="Earn $500 weekly from home! No experience needed. Send your CV and ID copy to hr.jobs@gmail.com today!"),
    "real": dict(BLANK, title="Senior Software Engineer", location="US, CA, San Francisco", employment_type="Full-time",
                 link="https://www.examplecorp.com/careers/senior-software-engineer", logo=True, questions=True,
                 company_profile="We are a growing software company with 200 employees building payments tools for clients worldwide.",
                 ad="Design and build backend services in Python. Work with product and design teams to ship features. Participate in code reviews and mentor junior engineers.",
                 requirements="5+ years of experience with Python, SQL and cloud platforms. Bachelor's degree in computer science or equivalent.",
                 benefits="Health insurance, 401k, flexible hours."),
}
