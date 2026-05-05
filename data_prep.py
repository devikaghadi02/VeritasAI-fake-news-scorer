import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

real_statements = [
    "The unemployment rate fell to 3.7 percent in November.",
    "Congress passed the bill with a bipartisan majority of 67 votes.",
    "The Federal Reserve raised interest rates by 0.25 percent.",
    "Scientists confirmed the vaccine is 94 percent effective in trials.",
    "The state budget deficit has grown to 1.2 billion dollars.",
]
mixed_statements = [
    "The new policy will reportedly save taxpayers millions over five years.",
    "Sources suggest the administration is considering major reforms.",
    "The proposed law could allegedly affect millions of Americans.",
    "Officials claim the project will create thousands of jobs.",
    "Analysts say the economy may recover faster than expected.",
]
fake_statements = [
    "SHOCKING: Government secretly funding radical agenda exposed!!!",
    "They don't want you to know the REAL truth about this cover-up!",
    "Mainstream media HIDING bombshell report about secret deals!!!",
    "BREAKING: Unbelievable scandal rocks the establishment overnight!",
    "The deep state is conspiring against ordinary Americans RIGHT NOW!",
]

def make_split(n=500):
    rows = []
    for _ in range(n // 3):
        rows.append({"statement": np.random.choice(real_statements)  + " " + str(np.random.randint(1000,9999)), "label": 0, "speaker": "politician", "subject": "economy", "context": "press conference"})
        rows.append({"statement": np.random.choice(mixed_statements) + " " + str(np.random.randint(1000,9999)), "label": 1, "speaker": "analyst",    "subject": "policy",  "context": "interview"})
        rows.append({"statement": np.random.choice(fake_statements)  + " " + str(np.random.randint(1000,9999)), "label": 2, "speaker": "unknown",     "subject": "politics","context": "social media"})
    return pd.DataFrame(rows)

train_df = make_split(900)
val_df   = make_split(150)
test_df  = make_split(150)

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv",     index=False)
test_df.to_csv("data/test.csv",   index=False)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print("Done!")