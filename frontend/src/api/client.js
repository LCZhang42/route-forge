import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const generatePath = async (grade, options = {}) => {
  const {
    temperature = 1.0,
    min_holds = 3,
    max_holds = 30,
    use_constraints = true,
  } = options

  const response = await apiClient.post('/api/generate', {
    grade,
    temperature,
    min_holds,
    max_holds,
    use_constraints,
  })
  return response.data
}

export const getGrades = async () => {
  const response = await apiClient.get('/api/grades')
  return response.data.grades
}

export const healthCheck = async () => {
  const response = await apiClient.get('/api/health')
  return response.data
}

export default apiClient
