import os
import json
import random
from urllib.parse import quote_plus
from datetime import datetime

from sqlalchemy import create_engine, Column, String, Float, Integer, TIMESTAMP, MetaData, Table
from sqlalchemy.sql import func
from faker import Faker
from dotenv import load_dotenv

# ------------------ Load environment ------------------
load_dotenv(override=True)

DB_USER = os.getenv("PG_USER")
DB_PASSWORD = quote_plus(os.getenv("PG_PASSWORD"))
DB_HOST = os.getenv("PG_HOST")
DB_PORT = os.getenv("PG_PORT")
DB_NAME = os.getenv("PG_NAME")

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
metadata = MetaData()
fake = Faker()

# ------------------ Define tables ------------------
customers = Table(
    'customers', metadata,
    Column('customer_id', String, primary_key=True),
    Column('name', String),
    Column('email', String),
    Column('phone', String),
    Column('churn_risk', Float)
)

offers = Table(
    'offers', metadata,
    Column('offer_id', Integer, primary_key=True, autoincrement=True),
    Column('offer_name', String),
    Column('offer_description', String),
    Column('categories', String),  # JSON string
    Column('created_at', TIMESTAMP, server_default=func.now())
)

# ------------------ Create tables (drop if needed) ------------------
metadata.drop_all(engine)  # optional: drop existing tables
metadata.create_all(engine)

# ------------------ Predefined generic offers ------------------
offer_templates = [
    {"name": "Premium Data Booster", "description": "Add 20GB extra data per month for 3 months.", "categories": ["data","mobile","internet_speed"]},
    {"name": "Bill Saver Discount", "description": "25% discount on monthly bill for 6 months.", "categories": ["billing","cost_saving","loyalty"]},
    {"name": "SpeedMax Upgrade", "description": "Ultra-fast internet upgrade for 2 months at no extra cost.", "categories": ["internet_speed","performance","premium"]},
    {"name": "Loyalty Rewards Credit", "description": "$75 credit for loyalty.", "categories": ["billing","loyalty","customer_appreciation"]},
    {"name": "Entertainment Bundle", "description": "3 months free streaming subscriptions.", "categories": ["entertainment","bonus","loyalty"]}
]

# ------------------ 50 Customer IDs ------------------
customer_ids = [
    "9237-HQITU","9305-CDSKC","7892-POOKP","0280-XJGEX","6467-CHFZW","6047-YHPVI","5380-WJKOV",
    "8168-UQWWF","7760-OYPDY","9420-LOJKX","7495-OOKFY","1658-BYGOY","5698-BQJOH","5919-TMRGD",
    "9191-MYQKX","8637-XJIVR","4598-XLKNJ","0486-HECZI","4846-WHAFZ","5299-RULOA","0404-SWRVG",
    "4412-YLTKF","0390-DCFDQ","3874-EQOEP","0867-MKZVY","3376-BMGFE","1875-QIVME","2656-FMOKZ",
    "2070-FNEXE","9367-WXLCH","1918-ZBFQJ","2472-OVKUP","1285-OKIPP","7825-ECJRF","9408-SSNVZ",
    "8937-RDTHP","0094-OIFMO","9947-OTFQU","4629-NRXKX","3606-TWKGI","4385-GZQXV","5940-AHUHD",
    "6432-TWQLB","4484-GLZOU","9512-UIBFX","5583-SXDAG","3488-PGMQJ","7534-BFESC","6390-DSAZX",
    "8098-LLAZX"
]

# ------------------ Seed data ------------------
try:
    with engine.begin() as conn:  # transactional scope
        # Bulk insert customers
        customer_rows = [
            {
                "customer_id": cid,
                "name": fake.name(),
                "email": fake.email(),
                "phone": fake.phone_number(),
                "churn_risk": round(random.uniform(0,1),2)
            }
            for cid in customer_ids
        ]
        conn.execute(customers.insert(), customer_rows)

        # Bulk insert offers
        offer_rows = [
            {
                "offer_name": offer["name"],
                "offer_description": offer["description"],
                "categories": json.dumps(offer["categories"])
            }
            for offer in offer_templates
        ]
        conn.execute(offers.insert(), offer_rows)

    print("✅ Customers and generic offers inserted successfully!")

except Exception as e:
    print("❌ Error inserting data:", e)
