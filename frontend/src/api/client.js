import axios from 'axios'

export const API_BASE_URL = import.meta.env.VITE_API_URL
  || 'https://lczhang42--climb-path-phase3-api.modal.run'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120_000,
})

let capabilitiesRequest = null

export const getModelCapabilities = async () => {
  if (!capabilitiesRequest) {
    capabilitiesRequest = apiClient
      .get('/api/v1/model')
      .then((response) => response.data)
      .catch((error) => {
        capabilitiesRequest = null
        throw error
      })
  }
  return capabilitiesRequest
}

export const generateRoutes = async (payload) => {
  const response = await apiClient.post('/api/v1/routes/generate', payload)
  return response.data
}

export const healthCheck = async () => {
  const response = await apiClient.get('/api/v1/health/ready')
  return response.data
}

export function getApiErrorMessage(error) {
  if (error?.code === 'ECONNABORTED') {
    return 'The model took too long to wake up. Please try once more.'
  }
  const detail = error?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (!error?.response) {
    return 'The route service could not be reached. Check your connection and retry.'
  }
  return 'The route service could not complete this request.'
}

export default apiClient
