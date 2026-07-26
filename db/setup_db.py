"""
Creates a small sample analytics DB (SQLite) so the whole eval harness
is runnable end-to-end without needing your real warehouse connected yet.

Swap this out for a read-only connection / snapshot of your real DB once
you're ready -- everything downstream only assumes a sqlite3-compatible
connection and a schema description string.
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "analytics.db")

SCHEMA_SQL = """
CREATE TABLE regions (
    region_id INTEGER PRIMARY KEY,
    region_name TEXT NOT NULL
);

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    region_id INTEGER NOT NULL,
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,   -- ISO 'YYYY-MM-DD'
    quantity INTEGER NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
"""

SEED_SQL = """
INSERT INTO regions VALUES (1,'North America'),(2,'EMEA'),(3,'APAC');

INSERT INTO customers VALUES
 (1,'Acme Corp',1),(2,'Globex',2),(3,'Initech',1),
 (4,'Umbrella Ltd',3),(5,'Soylent Inc',2);

INSERT INTO products VALUES
 (1,'Widget A','Widgets',9.99),
 (2,'Widget B','Widgets',14.99),
 (3,'Gadget X','Gadgets',49.99),
 (4,'Gadget Y','Gadgets',79.99);

INSERT INTO orders (customer_id, product_id, order_date, quantity) VALUES
 (1,1,'2026-04-02',10),(1,3,'2026-04-10',2),
 (2,2,'2026-04-03',5), (2,4,'2026-05-01',1),
 (3,1,'2026-05-15',20),(3,3,'2026-05-20',3),
 (4,4,'2026-04-22',4), (4,2,'2026-05-02',8),
 (5,1,'2026-06-01',12),(5,3,'2026-06-05',6);
"""

SCHEMA_DESCRIPTION = """
Tables:
- regions(region_id, region_name)
- customers(customer_id, customer_name, region_id -> regions.region_id)
- products(product_id, product_name, category, unit_price)
- orders(order_id, customer_id -> customers.customer_id,
         product_id -> products.product_id, order_date, quantity)

Revenue for an order = orders.quantity * products.unit_price.
""".strip()


def build_db(path: str = DB_PATH, overwrite: bool = True):
    if overwrite and os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)
    conn.executescript(SEED_SQL)
    conn.commit()
    conn.close()
    return path


if __name__ == "__main__":
    build_db()
    print(f"Built sample DB at {DB_PATH}")
