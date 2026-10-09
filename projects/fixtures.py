"""Synthetic software fixtures, not real-world evaluation data."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    templates={"order_status":["where is my package","track my shipment","my delivery is late","show order progress","has my order shipped","find the delivery date","check parcel location","when will it arrive"],"refund":["please refund this purchase","return my money","cancel and refund order","I want a return","the item is damaged","send a refund request","how do I return this","refund my payment"],"greeting":["hello there","good morning team","hi support","nice to meet you","hey how are you","good evening","hello can we chat","hi everyone"]}
    samples=[]
    for intent, phrases in templates.items():
        for phrase in phrases:
            for suffix,sentiment in [("thank you this is great","positive"),("I am upset about this","negative"),("please advise","neutral")]:
                samples.append((f"{phrase} {suffix}",intent,sentiment))
    pd.DataFrame(samples,columns=["text","intent","sentiment"]).to_csv(root / "messages.csv",index=False)
    (root / "NOTICE.txt").write_text("SYNTHETIC SOFTWARE TEST FIXTURES. Not real-world evaluation data.", encoding="utf-8")
    return root

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    print(generate(parser.parse_args().destination))
