/**
 * API Client for VeroAllarme Backend
 * Handles all HTTP requests to the FastAPI backend
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const alertsAPI = {
  // Get all alerts
  getAlerts: (params = {}) => 
    apiClient.get('/alerts', { params }),
  
  // Get single alert by ID
  getAlert: (id) => 
    apiClient.get(`/alerts/${id}`),
  
  // Create new alert (from camera)
  createAlert: (data) => 
    apiClient.post('/alerts', data),
  
  // Submit feedback for alert
  submitFeedback: (alertId, feedback) => 
    apiClient.post('/feedback', { alert_id: alertId, ...feedback }),
};

export const heatmapAPI = {
  // Get heat map for camera
  getHeatmap: (cameraId) => 
    apiClient.get(`/heatmap/${cameraId}`),
  
  // Get heat map image
  getHeatmapImage: (cameraId) => 
    apiClient.get(`/heatmap/${cameraId}/image`, { responseType: 'blob' }),
};

export const masksAPI = {
  // Get masks for camera
  getMasks: (cameraId) => 
    apiClient.get(`/masks/${cameraId}`),
  
  // Create/Update mask
  saveMask: (cameraId, maskData) => 
    apiClient.post(`/masks/${cameraId}`, maskData),
  
  // Delete mask
  deleteMask: (maskId) => 
    apiClient.delete(`/masks/${maskId}`),
};

export const statsAPI = {
  // Get dashboard statistics
  getStats: (period = '7d') => 
    apiClient.get('/stats', { params: { period } }),
};

export default apiClient;
