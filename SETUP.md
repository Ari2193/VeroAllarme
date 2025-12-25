# VeroAllarme - Quick Start Guide

## 🚀 Setup Instructions

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- Node.js 20+ (for frontend development)

### Quick Start with Docker

1. **Clone and setup:**
```bash
git clone <repository-url>
cd VeroAllarme
cp .env.example .env
```

2. **Download YOLO model:**
```bash
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
cd ..
```

3. **Start all services:**
```bash
docker-compose up -d
```

4. **Access the application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Flower (Celery monitoring): http://localhost:5555

### Local Development Setup

#### Backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

#### Frontend:
```bash
cd frontend
npm install
npm run dev
```

## 📁 Project Structure

```
VeroAllarme/
├── backend/
│   ├── api/              # FastAPI routes
│   ├── core/             # Algorithm stages
│   ├── models/           # Database models
│   ├── services/         # YOLO, visualization
│   ├── config.py         # Configuration
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/          # API client
│   │   ├── components/   # React components
│   │   └── views/        # Pages
│   ├── Dockerfile
│   ├── package.json
│   └── vite.config.js
├── data/
│   ├── images/           # Alert images
│   ├── masks/            # Region masks
│   └── heatmaps/         # Heat maps
├── models/               # YOLO weights
├── docker-compose.yml
├── .env.example
└── README.md
```

## 🔧 Configuration

Edit `.env` file to customize:
- Database connection
- Redis URL
- YOLO device (cpu/cuda)
- Motion detection thresholds
- Heat map settings

## 📊 Database Setup

Database is automatically initialized by Docker. For migrations:

```bash
cd backend
alembic init migrations
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head
```

## 🧪 Testing

```bash
cd backend
pytest tests/
```

## 📝 Next Steps

1. Implement core algorithm modules in `backend/core/`
2. Create database models in `backend/models/`
3. Build API endpoints in `backend/api/`
4. Develop frontend components in `frontend/src/`
5. Configure camera webhook integration

## 🐛 Troubleshooting

**YOLO model not found:**
```bash
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

**Port already in use:**
```bash
# Change ports in docker-compose.yml or .env
```

**Permission errors:**
```bash
sudo chown -R $USER:$USER data/
```

## 📚 Documentation

- Backend API: http://localhost:8000/docs
- Full README: [README.md](README.md)
- Hebrew README: [README.he.md](README.he.md)

---

Built for hackathon excellence! 🏆
