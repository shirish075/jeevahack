import sqlite3 from 'sqlite3';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const dbPath = join(__dirname, '../../subscription_management.db');

const db = new sqlite3.Database(dbPath);

// Drop existing tables
const dropTables = `
DROP TABLE IF EXISTS audit_logs;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS discounts;
DROP TABLE IF EXISTS subscription_plans;
DROP TABLE IF EXISTS users;
`;

// Create tables
const createTables = `
-- Users table
CREATE TABLE users (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4))||'-'||hex(randomblob(2))||'-'||hex(randomblob(2))||'-'||hex(randomblob(2))||'-'||hex(randomblob(6)))),
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'EndUser',
    created_at TEXT DEFAULT (CURRENT_TIMESTAMP),
    updated_at TEXT DEFAULT (CURRENT_TIMESTAMP)
);

-- Subscription plans table
CREATE TABLE subscription_plans (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4))||'-'||hex(randomblob(2))||'-'||hex(randomblob(2))||'-'||hex(randomblob(2))||'-'||hex(randomblob(6)))),
    name TEXT NOT NULL,
    description TEXT,
    product_type TEXT NOT NULL,
    price REAL NOT NULL,
    data_quota INTEGER NOT NULL,
    duration_days INTEGER NOT NULL,
    is_active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (CURRENT_TIMESTAMP),
    updated_at TEXT DEFAULT (CURRENT_TIMESTAMP)
);

-- Discounts table
CREATE TABLE discounts (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4))||'-'||hex(randomblob(2))||'-'||hex(randomblob(2))||'-'||hex(randomblob(2))||'-'||hex(randomblob(6)))),
    name TEXT NOT NULL,
    description TEXT,
    discount_percent REAL NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    plan_id TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (CURRENT_TIMESTAMP),
    updated_at TEXT DEFAULT (CURRENT_TIMESTAMP),
    FOREIGN KEY (plan_id) REFERENCES subscription_plans(id)
);

-- Subscriptions table
CREATE TABLE subscriptions (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4))||'-'||hex(randomblob(2))||'-'||hex(randomblob(2))||'-'||hex(randomblob(2))||'-'||hex(randomblob(6)))),
    user_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    auto_renew INTEGER DEFAULT 1,
    data_used INTEGER DEFAULT 0,
    discount_id TEXT,
    created_at TEXT DEFAULT (CURRENT_TIMESTAMP),
    updated_at TEXT DEFAULT (CURRENT_TIMESTAMP),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (plan_id) REFERENCES subscription_plans(id),
    FOREIGN KEY (discount_id) REFERENCES discounts(id)
);

-- Audit logs table
CREATE TABLE audit_logs (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4))||'-'||hex(randomblob(2))||'-'||hex(randomblob(2))||'-'||hex(randomblob(2))||'-'||hex(randomblob(6)))),
    user_id TEXT,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    details TEXT,
    timestamp TEXT DEFAULT (CURRENT_TIMESTAMP),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
`;

db.serialize(() => {
  console.log('🗄️  Creating subscription management database...');
  
  db.exec(dropTables, (err) => {
    if (err) {
      console.error('Error dropping tables:', err);
    } else {
      console.log('✅ Old tables dropped');
    }
  });

  db.exec(createTables, (err) => {
    if (err) {
      console.error('Error creating tables:', err);
    } else {
      console.log('✅ Tables created successfully');
    }
  });
});

db.close((err) => {
  if (err) {
    console.error('Error closing database:', err);
  } else {
    console.log('✅ Database migration completed');
  }
});