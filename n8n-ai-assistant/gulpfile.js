const gulp = require('gulp');

gulp.task('build:icons', () => {
	return gulp.src('assets/**/*').pipe(gulp.dest('dist/'));
});

gulp.task('default', gulp.series('build:icons'));