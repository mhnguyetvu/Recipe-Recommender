#!/bin/bash

# Recipe Recommender - Monitoring Setup Script

echo "=========================================="
echo "Recipe Recommender - Monitoring Setup"
echo "=========================================="
echo ""

# Check Docker
echo "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    echo "   Visit: https://www.docker.com/get-started"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker found: $(docker --version)"
echo "✅ Docker Compose found: $(docker-compose --version)"
echo ""

# Build and start services
echo "Building and starting services..."
echo "This may take a few minutes on first run..."
echo ""

docker-compose build
docker-compose up -d

echo ""
echo "=========================================="
echo "✅ Services Started Successfully!"
echo "=========================================="
echo ""
echo "📡 API:        http://localhost:2222"
echo "📊 Prometheus: http://localhost:9090"
echo "📈 Grafana:    http://localhost:3000"
echo ""
echo "Grafana Login:"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "=========================================="
echo "Useful Commands:"
echo "=========================================="
echo ""
echo "View logs:     docker-compose logs -f api"
echo "Stop services: docker-compose down"
echo "Restart:       docker-compose restart"
echo ""
echo "For more info, see MONITORING.md"
echo ""
