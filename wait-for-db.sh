#!/usr/bin/env bash
set -e

echo "⏳ Waiting for database to be ready..."
until pg_isready -h db -p 5432 -U bot; do
  sleep 1
done
echo "✅ Database is ready. Applying migrations..."
alembic -c /app/alembic.ini upgrade head
echo "✅ Migrations complete."
