const GRADES = ['6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C', '7C+', '8A', '8A+', '8B', '8B+']

function GradeSelector({ selectedGrade, onGradeChange }) {
  return (
    <div className="bg-slate-800 rounded-lg p-6 shadow-2xl">
      <h3 className="text-xl font-bold text-white mb-4">Select Difficulty</h3>
      <div className="grid grid-cols-2 gap-2">
        {GRADES.map((grade) => (
          <button
            key={grade}
            onClick={() => onGradeChange(grade)}
            className={`px-4 py-3 rounded-lg font-semibold transition-all ${
              selectedGrade === grade
                ? 'bg-blue-500 text-white shadow-lg scale-105'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            {grade}
          </button>
        ))}
      </div>
      <div className="mt-4 p-3 bg-slate-700 rounded-lg">
        <p className="text-sm text-slate-300">
          <span className="font-semibold text-white">Selected:</span> {selectedGrade}
        </p>
      </div>
    </div>
  )
}

export default GradeSelector
