function GradeSelector({ grades, selectedGrade, onGradeChange, disabled }) {
  return (
    <fieldset className="grade-fieldset" disabled={disabled}>
      <legend>Target grade</legend>
      <div className="grade-grid" role="radiogroup" aria-label="Target grade">
        {grades.map((grade) => (
          <button
            key={grade}
            type="button"
            role="radio"
            aria-checked={selectedGrade === grade}
            className={selectedGrade === grade ? 'grade-chip is-active' : 'grade-chip'}
            onClick={() => onGradeChange(grade)}
          >
            {grade}
          </button>
        ))}
      </div>
    </fieldset>
  )
}

export default GradeSelector
