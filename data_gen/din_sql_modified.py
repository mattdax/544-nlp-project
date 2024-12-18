import os
import time
import json
import multiprocessing
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
import tiktoken
from openai import OpenAI
import argparse
import re
from multiprocessing import Manager
import random
# Prompts

system_prompt_sql = """

# You are an SQLite expert. Your task is to generate both reasoning chains and the corresponding SQLite query given a user's question and relevant schema information.
# Always follow the instructions provided to generate the reasoning chains and SQLite in a structured and coherent manner

# Use ONLY THE SCHEMA LINKS to generate the SQLite query.\n

# Provide your reasoning chains step by step, following these steps:\n
1. Sequential Structure (Determine the order of SQL clauses: SELECT, FROM, JOIN, GROUP BY, ORDER BY, etc.)\n
2. Condition Structure (Apply filtering using WHERE or HAVING clauses to define specific conditions)\n
3. Join Structure (Use JOIN clauses to combine tables based on shared keys or relationships)\n
4. Aggregation Structure (Use aggregate functions like COUNT, SUM, AVG, etc., to summarize data)\n

##
Enclose your reasoning chains within <chains> and </chains> tags.\n
After your reasoning, output the final SQLite query enclosed within <SQL> and </SQL> tags.
Do not include any additional text outside these tags.\n
Do not include the SQLite query in the Reasoning; keep them separate' \n

# ***LEARN FROM THE EXAMPLES BELOW. PAY ATTENTION TO THE SCHEMA LINKS***.

"""

EASY_PROMPT = f'''

{system_prompt_sql}

Examples:

# Example 1
# User question
Find the buildings which have rooms with capacity more than 50.

# Relevant schema
Schema_links: [classroom.building,classroom.capacity,50]

# Example 1 output
<chains>
1. Sequential Structure: Start with the SELECT clause to specify the building names for rooms with a capacity greater than 50.
2. Condition Structure: Use a WHERE clause to filter rooms where "classroom.capacity" is more than 50.
3. Join Structure: No JOIN is required, as all necessary information is in the "classroom" table.
4. Aggregation Structure: No aggregation is needed, as we are simply retrieving buildings that meet the specified capacity condition.
</chains>
<SQL>
SELECT classroom.building
FROM classroom
WHERE classroom.capacity > 50;
</SQL>

# Example 2
# User question
Find the room number of the rooms which can sit 50 to 100 students and their buildings.

# Relevant schema
Schema_links: [classroom.building,classroom.room_number,classroom.capacity,50,100]

# Example 2 output
<chains>
1. Sequential Structure: Start with the SELECT clause to specify the room number and building of rooms with capacities in the range specified.
2. Condition Structure: Use a WHERE clause to filter rooms where "classroom.capacity" is between 50 and 100.
3. Join Structure: No JOIN is required since all necessary information is in the "classroom" table.
4. Aggregation Structure: No aggregation is needed, as we are simply retrieving room numbers and buildings that meet the specified capacity condition.
</chains>
<SQL>
SELECT classroom.room_number, classroom.building
FROM classroom
WHERE classroom.capacity BETWEEN 50 AND 100;
</SQL>

# Example 3
# User question
Give the name of the student in the History department with the most credits.

# Relevant schema
Schema_links: [student.name,student.dept_name,student.tot_cred,History]

# Example 3 output
<chains>
1. Sequential Structure: Start with the SELECT clause to retrieve the "student.name" of the student with the most credits in the History department.
2. Condition Structure: Use a WHERE clause to filter for students in the "History" department.
3. Join Structure: No JOIN is needed since all relevant information is available in the "student" table.
4. Aggregation Structure: Use ORDER BY in descending order on "student.tot_cred" to rank students by credits and apply LIMIT 1 to select the student with the highest total credits in the History department.
</chains>
<SQL>
SELECT student.name
FROM student
WHERE student.dept_name = 'History'
ORDER BY student.tot_cred DESC
LIMIT 1;
</SQL>

# Example 4
# User question
Find the total budgets of the Marketing or Finance department.

# Relevant schema
Schema_links: [department.budget,department.dept_name,Marketing,Finance]

# Example 4 output
<chains>
1. Sequential Structure: Begin with the SELECT clause to specify the required field. Since the question asks for "total budgets," an aggregation function (SUM) will be used on the "department.budget" field.
2. Condition Structure: Apply a WHERE clause to filter for the specific departments mentioned, i.e., "Marketing" or "Finance."
3. Join Structure: No JOIN is needed here, as the query only involves the "department" table.
4. Aggregation Structure: Use SUM to aggregate the budget values for the specified departments, providing the total budget for each.
</chains>
<SQL>
SELECT SUM(department.budget)
FROM department
WHERE department.dept_name = 'Marketing' OR department.dept_name = 'Finance';
</SQL>


# Example 5
# User question
Find the department name of the instructor whose name contains 'Soisalon'.

# Relevant schema
Schema_links: [instructor.dept_name,instructor.name,Soisalon]

# Example 5 output
<chains>
1. Sequential Structure: Start with the SELECT clause to retrieve the "instructor.dept_name" of the instructor whose name contains 'Soisalon'.
2. Condition Structure: Use a WHERE clause with the LIKE operator to filter for instructor names that contain 'Soisalon'.
3. Join Structure: No JOIN is necessary since all relevant information is in the "instructor" table.
4. Aggregation Structure: No aggregation is needed as we are retrieving a single field based on a specific condition.
</chains>
<SQL>
SELECT instructor.dept_name
FROM instructor
WHERE instructor.name LIKE '%Soisalon%';
</SQL>


# Example 6
# User question
What is the name of the department with the most credits?

# Relevant schema
Schema_links: [course.dept_name,course.credits]

# Example 6 output
<chains>
1. Sequential Structure: Start with the SELECT clause to retrieve the "course.dept_name" of the department with the most credits.
2. Condition Structure: No specific filtering condition is needed since we are looking for the department with the highest total credits.
3. Join Structure: No JOIN is required, as the information is available in the "course" table.
4. Aggregation Structure: Use SUM to calculate the total credits per department, then use ORDER BY in descending order on the summed credits to rank departments by total credits. Apply LIMIT 1 to select the department with the highest total credits.
</chains>
<SQL>
SELECT course.dept_name
FROM course
GROUP BY course.dept_name
ORDER BY SUM(course.credits) DESC
LIMIT 1;
</SQL>

# Example 7
# User question
How many instructors teach a course in the Spring of 2010?

# Relevant schema
Schema_links: [teaches.ID,teaches.semester,teaches.YEAR,Spring,2010]

# Example 7 output
<chains>
1. Sequential Structure: Begin with the SELECT clause to count the distinct instructor IDs, as the query asks for the number of instructors who taught a course.
2. Condition Structure: Use a WHERE clause to filter records for the "Spring" semester and the year "2010".
3. Join Structure: No JOIN is necessary since the required information is available within the "teaches" table.
4. Aggregation Structure: Use COUNT with DISTINCT on "teaches.ID" to count the unique instructors who meet the specified conditions.
</chains>
<SQL>
SELECT COUNT(DISTINCT teaches.ID)
FROM teaches
WHERE teaches.semester = 'Spring' AND teaches.year = 2010;
</SQL>


# Example 8
# User question
Find the name of the students and their department names sorted by their total credits in ascending order.

# Relevant schema
Schema_links: [student.name,student.dept_name,student.tot_cred]

# Example 8 output
<chains>
1. Sequential Structure: Begin with the SELECT clause to retrieve the fields "student.name" and "student.dept_name" along with "student.tot_cred" for sorting purposes.
2. Condition Structure: No specific filtering condition is required as we want all students and their department names.
3. Join Structure: No JOIN is needed since all necessary information is available in the "student" table.
4. Aggregation Structure: No aggregation function is needed here, but use ORDER BY to sort the results in ascending order based on "student.tot_cred".
</chains>
<SQL>
SELECT student.name, student.dept_name
FROM student
ORDER BY student.tot_cred ASC;
</SQL>

# Example 9
# User question
Find the year which offers the largest number of courses.

# Relevant schema
Schema_links: [SECTION.YEAR,SECTION.*]

# Example 9 output
<chains>
1. Sequential Structure: Begin with the SELECT clause to retrieve the "year" from the "SECTION" table since the question asks for the year with the largest number of courses.
2. Condition Structure: No specific conditions are required, as the query aims to find the year offering the maximum number of courses.
3. Join Structure: No JOIN is necessary because all the required data is contained within the "SECTION" table.
4. Aggregation Structure: Use the COUNT function to count the number of courses for each year. Use ORDER BY in descending order on the count and LIMIT 1 to retrieve the year with the largest number of courses.
</chains>
<SQL>
SELECT SECTION.year
FROM SECTION
GROUP BY SECTION.year
ORDER BY COUNT(*) DESC
LIMIT 1;
</SQL>


# Example 10
# User question
What are the names and average salaries for departments with average salary higher than 42000

# Relevant schema
Schema_links: [instructor.dept_name,instructor.salary,42000]

# Example 10 output
<chains>
1. Sequential Structure: Begin with the SELECT clause to specify the fields needed, which are the department name and the average salary.
2. Aggregation Structure: Use the AVG function to calculate the average salary for each department.
3. Grouping Structure: Use GROUP BY on the "instructor.dept_name" to calculate the average salary per department.
4. Condition Structure: Apply the HAVING clause to filter departments where the average salary is greater than 42000.
5. Join Structure: No JOIN is needed here, as the query only involves the "instructor" table.
</chains>
<SQL>
SELECT instructor.dept_name, AVG(instructor.salary) AS avg_salary
FROM instructor
GROUP BY instructor.dept_name
HAVING AVG(instructor.salary) > 42000;
</SQL>


# Example 11
# User question
How many rooms in each building have a capacity of over 50?

# Relevant schema
Schema_links: [classroom.*,classroom.building,classroom.capacity,50]

# Example 11 output
<chains>
1. Sequential Structure: Start with the SELECT clause to specify the fields needed, which are the building name and the count of rooms.
2. Condition Structure: Apply a WHERE clause to filter rooms with a capacity greater than 50.
3. Grouping Structure: Use GROUP BY on "classroom.building" to count rooms by each building.
4. Aggregation Structure: Use COUNT to calculate the number of rooms in each building that meet the capacity condition.
5. Join Structure: No JOIN is required since all needed data is within the "classroom" table.
</chains>
<SQL>
SELECT classroom.building, COUNT(*) AS room_count
FROM classroom
WHERE classroom.capacity > 50
GROUP BY classroom.building;
</SQL>


# Example 12
# User question
Find the names of the top 3 departments that provide the largest amount of courses?

# Relevant schema
Schema_links: [course.dept_name,course.*]

# Example 12 output
<chains>
1. Sequential Structure: Begin with the SELECT clause to specify only the department name, as the question does not require the course count.
2. Aggregation Structure: Use COUNT to calculate the number of courses offered by each department to determine the top departments.
3. Grouping Structure: Apply GROUP BY on "course.dept_name" to count courses for each department.
4. Ordering Structure: Use ORDER BY in descending order on the course count to rank departments by the number of courses they provide.
5. Limiting Structure: Use LIMIT 3 to retrieve only the names of the top 3 departments with the largest number of courses.
6. Join Structure: No JOIN is required since all necessary data is within the "course" table.
</chains>
<SQL>
SELECT course.dept_name
FROM course
GROUP BY course.dept_name
ORDER BY COUNT(*) DESC
LIMIT 3;
</SQL>


# Example 13
# User question
Find the maximum and average capacity among rooms in each building.

# Relevant schema
Schema_links: [classroom.building,classroom.capacity]

# Example 13 output
<chains>
1. Sequential Structure: Begin with the SELECT clause to specify the building and the required aggregate values, which are the maximum and average capacities.
2. Aggregation Structure: Use MAX and AVG functions to calculate the maximum and average capacity for rooms in each building.
3. Grouping Structure: Use GROUP BY on "classroom.building" to calculate these aggregate values for each building.
4. Join Structure: No JOIN is required since all necessary data is within the "classroom" table.
</chains>
<SQL>
SELECT classroom.building, MAX(classroom.capacity) AS max_capacity, AVG(classroom.capacity) AS avg_capacity
FROM classroom
GROUP BY classroom.building;
</SQL>

# Example 14
# User question
"Find the title of the course that is offered by more than one department.

# Relevant schema
Schema_links: [course.title]

# Exammple 14 output
<chains>
1. Sequential Structure: Begin with the SELECT clause to specify the required field, which is the course title.
2. Aggregation Structure: Use COUNT to determine the number of departments offering each course.
3. Grouping Structure: Use GROUP BY on "course.title" to aggregate the results by each course title.
4. Condition Structure: Apply the HAVING clause to filter for courses that are offered by more than one department, by setting the count to be greater than 1.
5. Join Structure: No JOIN is required since we are only interested in the "course" table.
</chains>
<SQL>
SELECT course.title
FROM course
GROUP BY course.title
HAVING COUNT(DISTINCT course.dept_name) > 1;
</SQL>

'''

MEDIUM_PROMPT = f"""

{system_prompt_sql}

# Example 1
# User question:
Find the total budgets of the Marketing or Finance department.
 
# Relevant schema:
Schema_links:
['department.budget', 'department.dept_name', 'Marketing', 'Finance']
 
# Output for Example 1
<chains>
1. Sequential Structure: Begin with the SELECT clause to specify the required field. Since the question asks for "total budgets," an aggregation function (SUM) will be used on the "department.budget" field.
2. Condition Structure: Apply a WHERE clause to filter for the specific departments mentioned, i.e., "Marketing" or "Finance."
3. Join Structure: No JOIN is needed here, as the query only involves the "department" table.
4. Aggregation Structure: Use SUM to aggregate the budget values for the specified departments, providing the total budget for each.
</chains>
<SQL>
SELECT SUM(department.budget)
FROM department
WHERE department.dept_name = 'Marketing' OR department.dept_name = 'Finance';
</SQL>

# Example 2
# User question:
Find the name and building of the department with the highest budget.
 
# Relevant schema:
Schema_links:
['department.budget', 'department.dept_name', 'department.building']
 
# Output for Example 2
<chains>
1. Sequential Structure: Start with the SELECT clause to retrieve the required fields, "department.dept_name" and "department.building."
2. Condition Structure: No specific filtering condition is needed here, as we are interested in finding the department with the highest budget.
3. Join Structure: No JOIN is necessary, as the information is available within the "department" table.
4. Aggregation Structure: Use ORDER BY in descending order on the "department.budget" to rank departments by budget. Use LIMIT 1 to select the department with the highest budget.
</chains>
<SQL>
SELECT department.dept_name, department.building
FROM department
ORDER BY department.budget DESC
LIMIT 1;
</SQL>

# Example 3
# User question:
What is the name and building of the departments whose budget is more than the average budget?
 
# Relevant schema:
Schema_links:
['department.budget', 'department.dept_name', 'department.building']
 
# Output for Example 3
<chains>
1. Sequential Structure: Begin with the SELECT clause to specify the fields we need, "department.dept_name" and "department.building."
2. Condition Structure: Apply a WHERE clause to filter departments with a budget above the average budget. We’ll use a subquery in the WHERE clause to calculate the average budget.
3. Join Structure: No JOIN is required since all the needed data is in the "department" table.
4. Aggregation Structure: Use AVG within a subquery to compute the average budget, then compare each department's budget against this average in the WHERE clause.
</chains>
<SQL>
SELECT department.dept_name, department.building
FROM department
WHERE department.budget > (SELECT AVG(department.budget) FROM department);
</SQL>

# Example 4
# User question:
Find the total number of students and total number of instructors for each department.
 
# Relevant schema:
Schema_links:
['department.dept_name = student.dept_name', 'student.id', 'department.dept_name = instructor.dept_name', 'instructor.id']
 
# Output for Example 4
<chains>
1. Sequential Structure: Begin with the SELECT clause to count the distinct "student.id" and "instructor.id" for each department. Also include "instructor.dept_name" in the SELECT clause to group results by department as specified in the ground truth.
2. Condition Structure: No filtering condition is specified, so we do not need a WHERE clause.
3. Join Structure: Use INNER JOIN clauses to connect the "department" table with both "student" and "instructor" tables based on "dept_name," ensuring that only departments with both students and instructors are included.
4. Aggregation Structure: Use COUNT with DISTINCT for both "student.id" and "instructor.id" to get the unique count of students and instructors per department, then GROUP BY "instructor.dept_name" to group results by each department.
</chains>
<SQL>
SELECT COUNT(DISTINCT student.id), 
       COUNT(DISTINCT instructor.id), 
       instructor.dept_name
FROM department AS T1
JOIN student AS T2 ON T1.dept_name = T2.dept_name
JOIN instructor AS T3 ON T1.dept_name = T3.dept_name
GROUP BY T3.dept_name;
</SQL>

# Example 5
# User question:
Find the title of courses that have two prerequisites?
 
# Relevant schema:
Schema_links:
['course.title', 'course.course_id = prereq.course_id']
 
# Output for Example 5
<chains>
1. Sequential Structure: Start with the SELECT clause to retrieve "course.title."
2. Condition Structure: Apply a HAVING clause to filter for courses that have exactly two prerequisites.
3. Join Structure: Use a JOIN between the "course" and "prereq" tables based on "course_id" to link each course with its prerequisites.
4. Aggregation Structure: Use COUNT on "prereq.course_id" to count the number of prerequisites per course, then filter with HAVING to select courses with exactly two prerequisites.
</chains>
<SQL>
SELECT course.title
FROM course
JOIN prereq ON course.course_id = prereq.course_id
GROUP BY course.course_id
HAVING COUNT(prereq.course_id) = 2;
</SQL>

# Example 6
# User question:
Find the name of students who took any class in the years of 2009 and 2010.
 
# Relevant schema:
Schema_links:
['student.name', 'student.id = takes.id', 'takes.YEAR', '2009', '2010']
 
# Output for Example 6
<chains>
1. Sequential Structure: Begin with the SELECT clause to retrieve the distinct names of students who took classes in the specified years.
2. Condition Structure: Use a WHERE clause to filter the records where "takes.year" is either 2009 or 2010.
3. Join Structure: Use a JOIN between the "student" and "takes" tables based on "student.id" and "takes.id" to associate each student with their class records.
4. Aggregation Structure: No aggregation is needed here, but DISTINCT is applied to ensure unique student names are returned.
</chains>
<SQL>
SELECT DISTINCT student.name
FROM student
JOIN takes ON student.id = takes.id
WHERE takes.year = 2009 OR takes.year = 2010;
</SQL>

# Example 7
# User question:
list in alphabetic order all course names and their instructors' names in year 2008.
 
# Relevant schema:
Schema_links:
['course.title', 'course.course_id = teaches.course_id', 'teaches.id = instructor.id', 'instructor.name', 'teaches.year', '2008']
 
# Output for Example 7
<chains>
1. Sequential Structure: Start with the SELECT clause to retrieve "course.title" and "instructor.name."
2. Condition Structure: Use a WHERE clause to filter for records where "teaches.year" is 2008.
3. Join Structure: Use JOINs to connect "course" with "teaches" on "course_id" and "teaches" with "instructor" on "instructor.id" to link each course with its respective instructor.
4. Aggregation Structure: No aggregation is needed, but apply ORDER BY to list results in alphabetical order by course title.
</chains>
<SQL>
SELECT course.title, instructor.name
FROM course
JOIN teaches ON course.course_id = teaches.course_id
JOIN instructor ON teaches.id = instructor.id
WHERE teaches.year = 2008
ORDER BY course.title ASC;
</SQL>

"""

HARD_PROMPT = f"""

{system_prompt_sql}


# Example 1
# User question:
Find the title of courses that have two prerequisites?
 
# Relevant schema:
Schema_links:
['course.title', 'course.course_id = prereq.course_id']
 
# Output for Example 1
<chains>
1. Sequential Structure: Start by selecting the "title" field from the "course" table, as the goal is to retrieve course titles.
2. Join Structure: Since we need information on prerequisites, join the "course" table with the "prereq" table on the "course_id" field to connect courses to their prerequisites.
3. Aggregation Structure: Use GROUP BY to group records by "course_id" in the "prereq" table, allowing us to count prerequisites for each course.
4. Condition Structure: Apply a HAVING clause to filter groups where the count of prerequisites is exactly 2, ensuring we only get courses with two prerequisites.
</chains>
<SQL>
SELECT course.title 
FROM course 
JOIN prereq ON course.course_id = prereq.course_id 
GROUP BY course.course_id 
HAVING COUNT(prereq.*) = 2;
</SQL>

# Example 2
# User question:
Find the name and building of the department with the highest budget.
 
# Relevant schema:
Schema_links:
['department.dept_name', 'department.building', 'department.budget']
 
# Output for Example 2
<chains>
1. Sequential Structure: Begin with SELECT clause to retrieve specific columns, followed by the FROM clause to specify the table source, and then use ORDER BY to sort by budget in descending order.
2. Condition Structure: No specific condition (WHERE clause) is needed as we are interested in the department with the highest budget.
3. Join Structure: Not required, as we only need data from the department table.
4. Aggregation Structure: Although no aggregate function is directly used, sorting by budget and limiting to the top result effectively gives us the department with the highest budget.
</chains>
<SQL>
SELECT dept_name, building
FROM department
ORDER BY budget DESC
LIMIT 1;
</SQL>

# Example 3
# User question:
Find the title, credit, and department name of courses that have more than one prerequisites?
 
# Relevant schema:
Schema_links:
['course.title', 'course.credits', 'course.dept_name', 'course.course_id = prereq.course_id']
 
# Output for Example 3
<chains>
1. Sequential Structure: Begin with the SELECT clause to retrieve the title, credits, and department name of courses. Use the FROM clause to specify the course table and JOIN it with the prereq table on the course_id to relate courses with their prerequisites. Apply GROUP BY on course_id to count prerequisites per course, and use the HAVING clause to filter courses with more than one prerequisite.
2. Condition Structure: Use the HAVING clause with COUNT(*) to filter for courses having more than one prerequisite.
3. Join Structure: Use an INNER JOIN between the course and prereq tables based on course_id to identify courses and their prerequisites.
4. Aggregation Structure: Use COUNT(*) within HAVING to aggregate the number of prerequisites per course and filter out courses with fewer than two prerequisites.
</chains>
<SQL>
SELECT course.title, course.credits, course.dept_name
FROM course
JOIN prereq ON course.course_id = prereq.course_id
GROUP BY course.course_id, course.title, course.credits, course.dept_name
HAVING COUNT(prereq.course_id) > 1;
</SQL>

# Example 4
# User question:
Give the name and building of the departments with greater than average budget.
 
# Relevant schema:
Schema_links:
['department.dept_name', 'department.building', 'department.budget']
 
# Output for Example 4
<chains>
1. Sequential Structure: Start with the SELECT clause to retrieve the department name and building. Use the FROM clause to specify the department table. Apply a WHERE clause to filter departments based on their budget compared to the average budget.
2. Condition Structure: Use the WHERE clause to select only those departments whose budget is greater than the calculated average budget.
3. Join Structure: Not needed, as we only need data from the department table.
4. Aggregation Structure: Use a subquery within the WHERE clause to calculate the average budget of all departments, filtering out departments whose budget is below this value.
</chains>
<SQL>
SELECT dept_name, building
FROM department
WHERE budget > (SELECT AVG(budget) FROM department);
</SQL>

# Example 5
# User question:
Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010.
 
# Relevant schema:
Schema_links:
['teaches.id', 'teaches.semester', 'teaches.YEAR', 'Fall', '2009', 'Spring', '2010']
 
# Output for Example 5
<chains>
1. Sequential Structure: Begin with the SELECT clause to retrieve instructor IDs. Use the FROM clause to specify the teaches table. Apply a WHERE clause to filter instructors who taught in Fall 2009, but not in Spring 2010.
2. Condition Structure: Use the WHERE clause with conditions to select instructors who taught in Fall 2009. To exclude instructors who taught in Spring 2010, use a subquery in the WHERE clause to identify IDs from Spring 2010 and ensure they are not included.
3. Join Structure: Not required, as all data is available within the teaches table.
4. Aggregation Structure: Not needed, as we are only retrieving IDs based on specified conditions without any aggregation.
</chains>
<SQL>
SELECT id
FROM teaches
WHERE semester = 'Fall' AND year = 2009
AND id NOT IN (SELECT id FROM teaches WHERE semester = 'Spring' AND year = 2010);
</SQL>

# Example 6
# User question:
Find the name of the courses that do not have any prerequisite?
 
# Relevant schema:
Schema_links:
['course.title', 'course.course_id']
 
# Output for Example 6
<chains>
1. Sequential Structure: Begin with the SELECT clause to retrieve the course titles from the course table. Use the FROM clause to specify the course table. Apply a WHERE clause to filter courses that do not have any prerequisites by ensuring that the course_id is not present in the prereq table.
2. Condition Structure: Use the WHERE clause with a subquery to exclude courses whose course_id appears in the prereq table, meaning they have prerequisites.
3. Join Structure: Not required, as we only need to filter based on the absence of matching course_id entries in the prereq table.
4. Aggregation Structure: Not needed, as no aggregation functions are required for this query.
</chains>
<SQL>
SELECT title
FROM course
WHERE course_id NOT IN (SELECT course_id FROM prereq);
</SQL>

# Example 7
# User question:
Find the salaries of all distinct instructors that are less than the largest salary.
 
# Relevant schema:
Schema_links:
['instructor.salary']
 
# Output for Example 7
<chains>
1. Sequential Structure: Start with the SELECT clause to retrieve distinct salary values from the instructor table. Use the FROM clause to specify the instructor table. Apply a WHERE clause to filter for salaries that are less than the maximum salary.
2. Condition Structure: Use the WHERE clause with a subquery to identify and exclude the largest salary. The subquery calculates the maximum salary from the instructor table, and the main query filters out salaries equal to or greater than this maximum.
3. Join Structure: Not required, as all data is from the instructor table.
4. Aggregation Structure: Use the MAX function in a subquery to find the largest salary, which is then used in the filtering condition.
</chains>
<SQL>
SELECT DISTINCT salary
FROM instructor
WHERE salary < (SELECT MAX(salary) FROM instructor);
</SQL>

# Example 8
# User question:
Find the names of students who have taken any course in the fall semester of year 2003.
 
# Relevant schema:
Schema_links:
['student.id', 'student.name', 'takes.id', 'takes.semester', 'fall', '2003']
 
# Output for Example 8
<chains>
1. Sequential Structure: Start with the SELECT clause to retrieve student names. Use the FROM clause to specify the student table and JOIN it with the takes table to link students with their course records. Apply a WHERE clause to filter records for courses taken in the fall semester of the year 2003.
2. Condition Structure: Use the WHERE clause to filter for entries where the semester is 'Fall' and the year is 2003.
3. Join Structure: Use an INNER JOIN between the student and takes tables based on the student ID to get the names of students who took courses in the specified semester and year.
4. Aggregation Structure: Not required, as we are retrieving specific records without any aggregation.
</chains>
<SQL>
SELECT student.name
FROM student
JOIN takes ON student.id = takes.id
WHERE takes.semester = 'Fall' AND takes.year = 2003;
</SQL>

# Example 9
# User question:
Find the minimum salary for the departments whose average salary is above the average payment of all instructors.
 
# Relevant schema:
Schema_links:
['instructor.salary', 'instructor.dept_name']
 
# Output for Example 9
<chains>
1. Sequential Structure: Start with the SELECT clause to retrieve the minimum salary and department name. Use the FROM clause to specify the instructor table. Apply a GROUP BY clause to calculate average salaries per department and filter departments based on the average salary condition.
2. Condition Structure: Use the HAVING clause to filter departments where the average salary is greater than the overall average salary of all instructors.
3. Join Structure: Not required, as all data is in the instructor table.
4. Aggregation Structure: Use AVG to calculate the department-specific average salary and the overall average salary. Use MIN to find the minimum salary in each qualifying department.
</chains>
<SQL>
SELECT dept_name, MIN(salary) AS min_salary
FROM instructor
GROUP BY dept_name
HAVING AVG(salary) > (SELECT AVG(salary) FROM instructor);
</SQL>

# Example 10
# User question:
What is the course title of the prerequisite of course Mobile Computing?
 
# Relevant schema:
Schema_links:
['course.title', 'course.course_id = prereq.course_id', 'prereq.prereq_id', 'course.title', 'Mobile Computing']
 
# Output for Example 10
<chains>
1. Sequential Structure: Begin with the SELECT clause to retrieve the course title of the prerequisite. Use the FROM clause to specify the course table and JOIN it with the prereq table on course_id. Apply a WHERE clause to identify the prerequisite of the course titled 'Mobile Computing'.
2. Condition Structure: Use the WHERE clause to filter for the course with the title 'Mobile Computing' and then retrieve the title of its prerequisite using the JOIN relationship.
3. Join Structure: Use an INNER JOIN between the course and prereq tables based on course_id to relate the course with its prerequisite.
4. Aggregation Structure: Not required, as we are retrieving specific records without any aggregation.
</chains>
<SQL>
SELECT course.title
FROM course
JOIN prereq ON course.course_id = prereq.prereq_id
WHERE prereq.course_id = (SELECT course_id FROM course WHERE title = 'Mobile Computing');
</SQL>

# Example 11
# User question:
Give the title and credits for the course that is taught in the classroom with the greatest capacity.
 
# Relevant schema:
Schema_links:
['classroom.capacity', 'classroom.building = SECTION.building', 'classroom.room_number = SECTION.room_number', 'course.title', 'course.credits', 'course.course_id = SECTION.course_id']
 
# Output for Example 11
<chains>
1. Sequential Structure: Begin with the SELECT clause to retrieve the title and credits of the course. Use the FROM clause to specify the course and classroom tables and join them with the section table on the building and room numbers, as well as the course_id. Apply a WHERE clause to filter for the room with the maximum capacity.
2. Condition Structure: Use a WHERE clause with a subquery to filter for the classroom with the maximum capacity.
3. Join Structure: Use JOINs between classroom, section, and course tables based on building, room_number, and course_id to connect course details with the room capacity.
4. Aggregation Structure: Use MAX to find the maximum capacity and filter based on this value to retrieve the relevant course details.
</chains>
<SQL>
SELECT course.title, course.credits
FROM course
JOIN section ON course.course_id = section.course_id
JOIN classroom ON classroom.building = section.building AND classroom.room_number = section.room_number
WHERE classroom.capacity = (SELECT MAX(capacity) FROM classroom);
</SQL>

"""
schema_linking_prompt = f"""Table advisor, columns = [*,s_ID,i_ID]
Table classroom, columns = [*,building,room_number,capacity]
Table course, columns = [*,course_id,title,dept_name,credits]
Table department, columns = [*,dept_name,building,budget]
Table instructor, columns = [*,ID,name,dept_name,salary]
Table prereq, columns = [*,course_id,prereq_id]
Table section, columns = [*,course_id,sec_id,semester,year,building,room_number,time_slot_id]
Table student, columns = [*,ID,name,dept_name,tot_cred]
Table takes, columns = [*,ID,course_id,sec_id,semester,year,grade]
Table teaches, columns = [*,ID,course_id,sec_id,semester,year]
Table time_slot, columns = [*,time_slot_id,day,start_hr,start_min,end_hr,end_min]
Foreign_keys = [course.dept_name = department.dept_name,instructor.dept_name = department.dept_name,section.building = classroom.building,section.room_number = classroom.room_number,section.course_id = course.course_id,teaches.ID = instructor.ID,teaches.course_id = section.course_id,teaches.sec_id = section.sec_id,teaches.semester = section.semester,teaches.year = section.year,student.dept_name = department.dept_name,takes.ID = student.ID,takes.course_id = section.course_id,takes.sec_id = section.sec_id,takes.semester = section.semester,takes.year = section.year,advisor.s_ID = student.ID,advisor.i_ID = instructor.ID,prereq.prereq_id = course.course_id,prereq.course_id = course.course_id]
Q: "Find the buildings which have rooms with capacity more than 50."
A: Let’s think step by step. In the question "Find the buildings which have rooms with capacity more than 50.", we are asked:
"the buildings which have rooms" so we need column = [classroom.capacity]
"rooms with capacity" so we need column = [classroom.building]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [50]. So the Schema_links are:
Schema_links: [classroom.building,classroom.capacity,50]

Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
Table head, columns = [*,head_ID,name,born_state,age]
Table management, columns = [*,department_ID,head_ID,temporary_acting]
Foreign_keys = [management.head_ID = head.head_ID,management.department_ID = department.Department_ID]
Q: "How many heads of the departments are older than 56 ?"
A: Let’s think step by step. In the question "How many heads of the departments are older than 56 ?", we are asked:
"How many heads of the departments" so we need column = [head.*]
"older" so we need column = [head.age]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [56]. So the Schema_links are:
Schema_links: [head.*,head.age,56]

Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
Table head, columns = [*,head_ID,name,born_state,age]
Table management, columns = [*,department_ID,head_ID,temporary_acting]
Foreign_keys = [management.head_ID = head.head_ID,management.department_ID = department.Department_ID]
Q: "what are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?"
A: Let’s think step by step. In the question "what are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?", we are asked:
"distinct creation years of the departments" so we need column = [department.Creation]
"departments managed by" so we need column = [management.department_ID]
"born in" so we need column = [head.born_state]
Based on the columns and tables, we need these Foreign_keys = [department.Department_ID = management.department_ID,management.head_ID = head.head_ID].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = ['Alabama']. So the Schema_links are:
Schema_links: [department.Creation,department.Department_ID = management.department_ID,head.head_ID = management.head_ID,head.born_state,'Alabama']

Table Addresses, columns = [*,address_id,line_1,line_2,city,zip_postcode,state_province_county,country]
Table Candidate_Assessments, columns = [*,candidate_id,qualification,assessment_date,asessment_outcome_code]
Table Candidates, columns = [*,candidate_id,candidate_details]
Table Courses, columns = [*,course_id,course_name,course_description,other_details]
Table People, columns = [*,person_id,first_name,middle_name,last_name,cell_mobile_number,email_address,login_name,password]
Table People_Addresses, columns = [*,person_address_id,person_id,address_id,date_from,date_to]
Table Student_Course_Attendance, columns = [*,student_id,course_id,date_of_attendance]
Table Student_Course_Registrations, columns = [*,student_id,course_id,registration_date]
Table Students, columns = [*,student_id,student_details]
Foreign_keys = [Students.student_id = People.person_id,People_Addresses.address_id = Addresses.address_id,People_Addresses.person_id = People.person_id,Student_Course_Registrations.course_id = Courses.course_id,Student_Course_Registrations.student_id = Students.student_id,Student_Course_Attendance.student_id = Student_Course_Registrations.student_id,Student_Course_Attendance.course_id = Student_Course_Registrations.course_id,Candidates.candidate_id = People.person_id,Candidate_Assessments.candidate_id = Candidates.candidate_id]
Q: "List the id of students who never attends courses?"
A: Let’s think step by step. In the question "List the id of students who never attends courses?", we are asked:
"id of students" so we need column = [Students.student_id]
"never attends courses" so we need column = [Student_Course_Attendance.student_id]
Based on the columns and tables, we need these Foreign_keys = [Students.student_id = Student_Course_Attendance.student_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = []. So the Schema_links are:
Schema_links: [Students.student_id = Student_Course_Attendance.student_id]

Table Country, columns = [*,id,name]
Table League, columns = [*,id,country_id,name]
Table Player, columns = [*,id,player_api_id,player_name,player_fifa_api_id,birthday,height,weight]
Table Player_Attributes, columns = [*,id,player_fifa_api_id,player_api_id,date,overall_rating,potential,preferred_foot,attacking_work_rate,defensive_work_rate,crossing,finishing,heading_accuracy,short_passing,volleys,dribbling,curve,free_kick_accuracy,long_passing,ball_control,acceleration,sprint_speed,agility,reactions,balance,shot_power,jumping,stamina,strength,long_shots,aggression,interceptions,positioning,vision,penalties,marking,standing_tackle,sliding_tackle,gk_diving,gk_handling,gk_kicking,gk_positioning,gk_reflexes]
Table Team, columns = [*,id,team_api_id,team_fifa_api_id,team_long_name,team_short_name]
Table Team_Attributes, columns = [*,id,team_fifa_api_id,team_api_id,date,buildUpPlaySpeed,buildUpPlaySpeedClass,buildUpPlayDribbling,buildUpPlayDribblingClass,buildUpPlayPassing,buildUpPlayPassingClass,buildUpPlayPositioningClass,chanceCreationPassing,chanceCreationPassingClass,chanceCreationCrossing,chanceCreationCrossingClass,chanceCreationShooting,chanceCreationShootingClass,chanceCreationPositioningClass,defencePressure,defencePressureClass,defenceAggression,defenceAggressionClass,defenceTeamWidth,defenceTeamWidthClass,defenceDefenderLineClass]
Table sqlite_sequence, columns = [*,name,seq]
Foreign_keys = [Player_Attributes.player_api_id = Player.player_api_id,Player_Attributes.player_fifa_api_id = Player.player_fifa_api_id,League.country_id = Country.id,Team_Attributes.team_api_id = Team.team_api_id,Team_Attributes.team_fifa_api_id = Team.team_fifa_api_id]
Q: "List the names of all left-footed players who have overall rating between 85 and 90."
A: Let’s think step by step. In the question "List the names of all left-footed players who have overall rating between 85 and 90.", we are asked:
"names of all left-footed players" so we need column = [Player.player_name,Player_Attributes.preferred_foot]
"players who have overall rating" so we need column = [Player_Attributes.overall_rating]
Based on the columns and tables, we need these Foreign_keys = [Player_Attributes.player_api_id = Player.player_api_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [left,85,90]. So the Schema_links are:
Schema_links: [Player.player_name,Player_Attributes.preferred_foot,Player_Attributes.overall_rating,Player_Attributes.player_api_id = Player.player_api_id,left,85,90]

Table advisor, columns = [*,s_ID,i_ID]
Table classroom, columns = [*,building,room_number,capacity]
Table course, columns = [*,course_id,title,dept_name,credits]
Table department, columns = [*,dept_name,building,budget]
Table instructor, columns = [*,ID,name,dept_name,salary]
Table prereq, columns = [*,course_id,prereq_id]
Table section, columns = [*,course_id,sec_id,semester,year,building,room_number,time_slot_id]
Table student, columns = [*,ID,name,dept_name,tot_cred]
Table takes, columns = [*,ID,course_id,sec_id,semester,year,grade]
Table teaches, columns = [*,ID,course_id,sec_id,semester,year]
Table time_slot, columns = [*,time_slot_id,day,start_hr,start_min,end_hr,end_min]
Foreign_keys = [course.dept_name = department.dept_name,instructor.dept_name = department.dept_name,section.building = classroom.building,section.room_number = classroom.room_number,section.course_id = course.course_id,teaches.ID = instructor.ID,teaches.course_id = section.course_id,teaches.sec_id = section.sec_id,teaches.semester = section.semester,teaches.year = section.year,student.dept_name = department.dept_name,takes.ID = student.ID,takes.course_id = section.course_id,takes.sec_id = section.sec_id,takes.semester = section.semester,takes.year = section.year,advisor.s_ID = student.ID,advisor.i_ID = instructor.ID,prereq.prereq_id = course.course_id,prereq.course_id = course.course_id]
Q: "Give the title of the course offered in Chandler during the Fall of 2010."
A: Let’s think step by step. In the question "Give the title of the course offered in Chandler during the Fall of 2010.", we are asked:
"title of the course" so we need column = [course.title]
"course offered in Chandler" so we need column = [SECTION.building]
"during the Fall" so we need column = [SECTION.semester]
"of 2010" so we need column = [SECTION.year]
Based on the columns and tables, we need these Foreign_keys = [course.course_id = SECTION.course_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [Chandler,Fall,2010]. So the Schema_links are:
Schema_links: [course.title,course.course_id = SECTION.course_id,SECTION.building,SECTION.year,SECTION.semester,Chandler,Fall,2010]

Table city, columns = [*,City_ID,Official_Name,Status,Area_km_2,Population,Census_Ranking]
Table competition_record, columns = [*,Competition_ID,Farm_ID,Rank]
Table farm, columns = [*,Farm_ID,Year,Total_Horses,Working_Horses,Total_Cattle,Oxen,Bulls,Cows,Pigs,Sheep_and_Goats]
Table farm_competition, columns = [*,Competition_ID,Year,Theme,Host_city_ID,Hosts]
Foreign_keys = [farm_competition.Host_city_ID = city.City_ID,competition_record.Farm_ID = farm.Farm_ID,competition_record.Competition_ID = farm_competition.Competition_ID]
Q: "Show the status of the city that has hosted the greatest number of competitions."
A: Let’s think step by step. In the question "Show the status of the city that has hosted the greatest number of competitions.", we are asked:
"the status of the city" so we need column = [city.Status]
"greatest number of competitions" so we need column = [farm_competition.*]
Based on the columns and tables, we need these Foreign_keys = [farm_competition.Host_city_ID = city.City_ID].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = []. So the Schema_links are:
Schema_links: [city.Status,farm_competition.Host_city_ID = city.City_ID,farm_competition.*]

Table advisor, columns = [*,s_ID,i_ID]
Table classroom, columns = [*,building,room_number,capacity]
Table course, columns = [*,course_id,title,dept_name,credits]
Table department, columns = [*,dept_name,building,budget]
Table instructor, columns = [*,ID,name,dept_name,salary]
Table prereq, columns = [*,course_id,prereq_id]
Table section, columns = [*,course_id,sec_id,semester,year,building,room_number,time_slot_id]
Table student, columns = [*,ID,name,dept_name,tot_cred]
Table takes, columns = [*,ID,course_id,sec_id,semester,year,grade]
Table teaches, columns = [*,ID,course_id,sec_id,semester,year]
Table time_slot, columns = [*,time_slot_id,day,start_hr,start_min,end_hr,end_min]
Foreign_keys = [course.dept_name = department.dept_name,instructor.dept_name = department.dept_name,section.building = classroom.building,section.room_number = classroom.room_number,section.course_id = course.course_id,teaches.ID = instructor.ID,teaches.course_id = section.course_id,teaches.sec_id = section.sec_id,teaches.semester = section.semester,teaches.year = section.year,student.dept_name = department.dept_name,takes.ID = student.ID,takes.course_id = section.course_id,takes.sec_id = section.sec_id,takes.semester = section.semester,takes.year = section.year,advisor.s_ID = student.ID,advisor.i_ID = instructor.ID,prereq.prereq_id = course.course_id,prereq.course_id = course.course_id]
Q: "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010."
A: Let’s think step by step. In the question "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010.", we are asked:
"id of instructors who taught " so we need column = [teaches.id]
"taught a class in" so we need column = [teaches.semester,teaches.year]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [Fall,2009,Spring,2010]. So the Schema_links are:
schema_links: [teaches.id,teaches.semester,teaches.year,Fall,2009,Spring,2010]

Table Accounts, columns = [*,account_id,customer_id,date_account_opened,account_name,other_account_details]
Table Customers, columns = [*,customer_id,customer_first_name,customer_middle_initial,customer_last_name,gender,email_address,login_name,login_password,phone_number,town_city,state_county_province,country]
Table Financial_Transactions, columns = [*,transaction_id,account_id,invoice_number,transaction_type,transaction_date,transaction_amount,transaction_comment,other_transaction_details]
Table Invoice_Line_Items, columns = [*,order_item_id,invoice_number,product_id,product_title,product_quantity,product_price,derived_product_cost,derived_vat_payable,derived_total_cost]
Table Invoices, columns = [*,invoice_number,order_id,invoice_date]
Table Order_Items, columns = [*,order_item_id,order_id,product_id,product_quantity,other_order_item_details]
Table Orders, columns = [*,order_id,customer_id,date_order_placed,order_details]
Table Product_Categories, columns = [*,production_type_code,product_type_description,vat_rating]
Table Products, columns = [*,product_id,parent_product_id,production_type_code,unit_price,product_name,product_color,product_size]
Foreign_keys = [Orders.customer_id = Customers.customer_id,Invoices.order_id = Orders.order_id,Accounts.customer_id = Customers.customer_id,Products.production_type_code = Product_Categories.production_type_code,Financial_Transactions.account_id = Accounts.account_id,Financial_Transactions.invoice_number = Invoices.invoice_number,Order_Items.order_id = Orders.order_id,Order_Items.product_id = Products.product_id,Invoice_Line_Items.product_id = Products.product_id,Invoice_Line_Items.invoice_number = Invoices.invoice_number,Invoice_Line_Items.order_item_id = Order_Items.order_item_id]
Q: "Show the id, the date of account opened, the account name, and other account detail for all accounts."
A: Let’s think step by step. In the question "Show the id, the date of account opened, the account name, and other account detail for all accounts.", we are asked:
"the id, the date of account opened, the account name, and other account detail for all accounts." so we need column = [Accounts.account_id,Accounts.account_name,Accounts.other_account_details,Accounts.date_account_opened]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = []. So the Schema_links are:
Schema_links: [Accounts.account_id,Accounts.account_name,Accounts.other_account_details,Accounts.date_account_opened]

Table city, columns = [*,City_ID,Official_Name,Status,Area_km_2,Population,Census_Ranking]
Table competition_record, columns = [*,Competition_ID,Farm_ID,Rank]
Table farm, columns = [*,Farm_ID,Year,Total_Horses,Working_Horses,Total_Cattle,Oxen,Bulls,Cows,Pigs,Sheep_and_Goats]
Table farm_competition, columns = [*,Competition_ID,Year,Theme,Host_city_ID,Hosts]
Foreign_keys = [farm_competition.Host_city_ID = city.City_ID,competition_record.Farm_ID = farm.Farm_ID,competition_record.Competition_ID = farm_competition.Competition_ID]
Q: "Show the status shared by cities with population bigger than 1500 and smaller than 500."
A: Let’s think step by step. In the question "Show the status shared by cities with population bigger than 1500 and smaller than 500.", we are asked:
"the status shared by cities" so we need column = [city.Status]
"cities with population" so we need column = [city.Population]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [1500,500]. So the Schema_links are:
Schema_links: [city.Status,city.Population,1500,500]

"""

classification_prompt = f"""Q: "Find the buildings which have rooms with capacity more than 50."
schema_links: [classroom.building,classroom.capacity,50]
A: Let’s think step by step. The SQL query for the question "Find the buildings which have rooms with capacity more than 50." needs these tables = [classroom], so we don't need JOIN.
Plus, it doesn't require nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we don't need JOIN and don't need nested queries, then the the SQL query can be classified as "EASY".
Label: "EASY"

Q: "What are the names of all instructors who advise students in the math depart sorted by total credits of the student."
schema_links: [advisor.i_id = instructor.id,advisor.s_id = student.id,instructor.name,student.dept_name,student.tot_cred,math]
A: Let’s think step by step. The SQL query for the question "What are the names of all instructors who advise students in the math depart sorted by total credits of the student." needs these tables = [advisor,instructor,student], so we need JOIN.
Plus, it doesn't need nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we need JOIN and don't need nested queries, then the the SQL query can be classified as "NON-NESTED".
Label: "NON-NESTED"

Q: "Find the room number of the rooms which can sit 50 to 100 students and their buildings."
schema_links: [classroom.building,classroom.room_number,classroom.capacity,50,100]
A: Let’s think step by step. The SQL query for the question "Find the room number of the rooms which can sit 50 to 100 students and their buildings." needs these tables = [classroom], so we don't need JOIN.
Plus, it doesn't require nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we don't need JOIN and don't need nested queries, then the the SQL query can be classified as "EASY".
Label: "EASY"

Q: "How many courses that do not have prerequisite?"
schema_links: [course.*,course.course_id = prereq.course_id]
A: Let’s think step by step. The SQL query for the question "How many courses that do not have prerequisite?" needs these tables = [course,prereq], so we need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Which courses have prerequisite?"].
So, we need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"

Q: "Find the title of course that is provided by both Statistics and Psychology departments."
schema_links: [course.title,course.dept_name,Statistics,Psychology]
A: Let’s think step by step. The SQL query for the question "Find the title of course that is provided by both Statistics and Psychology departments." needs these tables = [course], so we don't need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Find the titles of courses that is provided by Psychology departments"].
So, we don't need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"

Q: "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010."
schema_links: [teaches.id,teaches.semester,teaches.year,Fall,2009,Spring,2010]
A: Let’s think step by step. The SQL query for the question "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010." needs these tables = [teaches], so we don't need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Find the id of instructors who taught a class in Spring 2010"].
So, we don't need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"

Q: "Find the name of the department that offers the highest total credits?"
schema_links: [course.dept_name,course.credits]
A: Let’s think step by step. The SQL query for the question "Find the name of the department that offers the highest total credits?." needs these tables = [course], so we don't need JOIN.
Plus, it doesn't require nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we don't need JOIN and don't need nested queries, then the the SQL query can be classified as "EASY".
Label: "EASY"

Q: "What is the name of the instructor who advises the student with the greatest number of total credits?"
schema_links: [advisor.i_id = instructor.id,advisor.s_id = student.id,instructor.name,student.tot_cred ]
A: Let’s think step by step. The SQL query for the question "What is the name of the instructor who advises the student with the greatest number of total credits?" needs these tables = [advisor,instructor,student], so we need JOIN.
Plus, it doesn't need nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we need JOIN and don't need nested queries, then the the SQL query can be classified as "NON-NESTED".
Label: "NON-NESTED"

Q: "Find the total number of students and total number of instructors for each department."
schema_links = [department.dept_name = instructor.dept_name,student.id,student.dept_name = department.dept_name,instructor.id]
A: Let’s think step by step. The SQL query for the question "Find the total number of students and total number of instructors for each department." needs these tables = [department,instructor,student], so we need JOIN.
Plus, it doesn't need nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we need JOIN and don't need nested queries, then the the SQL query can be classified as "NON-NESTED".
Label: "NON-NESTED"

Q: "Give the name and building of the departments with greater than average budget."
schema_links: [department.budget,department.dept_name,department.building]
A: Let’s think step by step. The SQL query for the question "Give the name and building of the departments with greater than average budget." needs these tables = [department], so we don't need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["What is the average budget of the departments"].
So, we don't need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"

"""

# Initialize OpenAI API client
oai_api_key = os.getenv('OPENAI_API_KEY')
oai_client = OpenAI(api_key=oai_api_key)

@dataclass
class DatabaseSchema:
    tables: Dict[str, List[str]]
    foreign_keys: List[tuple]
    primary_keys: Dict[str, str]


class DINSQLPipeline:
    def __init__(self):
        """Initialize pipeline with OpenAI API key"""
        # chatgpt-4o-latest
        # gpt-4o-2024-11-20
        self.model = f"gpt-4o-2024-11-20"
        self.tokenizer = tiktoken.encoding_for_model(self.model)  # Use GPT-4o-mini
        self.client = oai_client
        self.current_results = {}
        self.spider_schema = None
        self.spider_primary = None
        self.spider_foreign = None

    def store_intermediate_result(self, step: str, data: Any):
        """Store intermediate result from any step"""
        self.current_results[step] = data

    def get_reasoning(self) -> str:
        """Get stored reasoning"""
        return self.current_results.get('sql_reasoning', '')

    def get_sql(self) -> str:
        """Get the initial SQL query prediction"""
        return self.current_results.get('generated_sql', '')

    def load_schema(self, schema_file: str) -> Dict[str, DatabaseSchema]:
        """Load database schemas from tables.json"""
        print(f"\nLoading schema from: {schema_file}")
        with open(schema_file) as f:
            schemas = json.load(f)

        print(f"Loaded {len(schemas)} database schemas")

        processed_schemas = {}
        for db in schemas:
            try:
                db_id = db['db_id']
                print(f"\nProcessing schema for database: {db_id}")

                # Process tables and columns
                tables = {}
                for i, table in enumerate(db['table_names_original']):
                    columns = []
                    for col_idx, (tab_idx, col_name) in enumerate(db['column_names_original']):
                        if tab_idx == i:
                            columns.append(col_name)
                    tables[table] = columns

                # Process foreign keys
                foreign_keys = []
                for fk in db['foreign_keys']:
                    try:
                        col1 = db['column_names_original'][fk[0]][1]
                        tab1 = db['table_names_original'][db['column_names_original'][fk[0]][0]]
                        col2 = db['column_names_original'][fk[1]][1]
                        tab2 = db['table_names_original'][db['column_names_original'][fk[1]][0]]
                        foreign_keys.append((f"{tab1}.{col1}", f"{tab2}.{col2}"))
                    except Exception as e:
                        print(f"Error processing foreign key {fk}: {str(e)}")

                # Process primary keys
                primary_keys = {}
                for pk in db['primary_keys']:
                    try:
                        table_idx = db['column_names_original'][pk][0]
                        col_name = db['column_names_original'][pk][1]
                        table_name = db['table_names_original'][table_idx]
                        primary_keys[table_name] = col_name
                    except Exception as e:
                        print(f"Error processing primary key {pk}: {str(e)}")

                processed_schemas[db_id] = DatabaseSchema(
                    tables=tables,
                    foreign_keys=foreign_keys,
                    primary_keys=primary_keys
                )

            except Exception as e:
                print(f"Error processing database {db.get('db_id', 'unknown')}: {str(e)}")
                continue

        return processed_schemas

    def creating_schema(self, DATASET_JSON):
        """Create schema DataFrames from tables.json"""
        print("\n=== Starting Schema Processing ===")
        print(f"Reading JSON from: {DATASET_JSON}")

        try:
            schema_df = pd.read_json(DATASET_JSON)
        except Exception as e:
            print(f"Error in initial schema reading: {str(e)}")
            return None, None, None

        try:
            schema_df = schema_df.drop(['column_names', 'table_names'], axis=1)
        except Exception as e:
            print(f"Warning in dropping columns: {str(e)}")

        schema = []
        f_keys = []
        p_keys = []

        for index, row in schema_df.iterrows():
            try:
                db_id = row['db_id']
                tables = row['table_names_original']
                col_names = row['column_names_original']
                col_types = row['column_types']
                foreign_keys = row['foreign_keys']
                primary_keys = row['primary_keys']

                # Process columns
                for col, col_type in zip(col_names, col_types):
                    index, col_name = col
                    if index == -1:
                        for table in tables:
                            schema.append([db_id, table, '*', 'text'])
                    else:
                        try:
                            table_name = tables[index]
                            schema.append([db_id, table_name, col_name, col_type])
                        except IndexError:
                            print(f"Warning: Invalid table index {index} for column {col_name}")
                            continue

                # Process primary keys
                for pk in primary_keys:
                    try:
                        table_idx, col_name = col_names[pk]
                        if table_idx >= 0:  # Skip the * columns
                            p_keys.append([db_id, tables[table_idx], col_name])
                    except IndexError:
                        print(f"Warning: Invalid primary key index {pk}")
                        continue

                # Process foreign keys
                for fk_pair in foreign_keys:
                    try:
                        fk1, fk2 = fk_pair
                        # Get first key info
                        tab1_idx, col1_name = col_names[fk1]
                        # Get second key info
                        tab2_idx, col2_name = col_names[fk2]

                        if tab1_idx >= 0 and tab2_idx >= 0:  # Skip the * columns
                            f_keys.append([
                                db_id,
                                tables[tab1_idx],
                                tables[tab2_idx],
                                col1_name,
                                col2_name
                            ])
                    except IndexError:
                        print(f"Warning: Invalid foreign key indices {fk_pair}")
                        continue

            except Exception as e:
                print(f"Error processing database {row.get('db_id', 'unknown')}: {str(e)}")
                print(f"Error details: {type(e).__name__}")
                continue

        print("\n=== Creating Final DataFrames ===")

        # Create DataFrames
        self.spider_schema = pd.DataFrame(schema,
                                          columns=['Database name', 'Table Name', 'Field Name', 'Type'])
        self.spider_primary = pd.DataFrame(p_keys,
                                           columns=['Database name', 'Table Name', 'Primary Key'])
        self.spider_foreign = pd.DataFrame(f_keys,
                                           columns=['Database name', 'First Table Name', 'Second Table Name',
                                                    'First Table Foreign Key', 'Second Table Foreign Key'])

        if len(schema) == 0:
            print("\nWARNING: No schema entries were created!")
        if len(p_keys) == 0:
            print("\nWARNING: No primary keys were found!")
        if len(f_keys) == 0:
            print("\nWARNING: No foreign keys were found!")

        return self.spider_schema, self.spider_primary, self.spider_foreign

    # Helper Functions Defined
    def find_fields_MYSQL_like(self, db_name):
        """Format fields in MySQL-like syntax"""
        df = self.spider_schema[self.spider_schema['Database name'] == db_name]
        df = df.groupby('Table Name')
        output = ""
        for name, group in df:
            output += "Table " + name + ', columns = ['
            for index, row in group.iterrows():
                output += row["Field Name"] + ','
            output = output[:-1]
            output += "]\n"
        return output

    def find_primary_keys_MYSQL_like(self, db_name):
        """Format primary keys in MySQL-like syntax"""
        df = self.spider_primary[self.spider_primary['Database name'] == db_name]
        output = "["
        for index, row in df.iterrows():
            output += row['Table Name'] + '.' + row['Primary Key'] + ','
        output = output[:-1] + "]" if output != "[" else "[]"
        return output

    def find_foreign_keys_MYSQL_like(self, db_name):
        """Format foreign keys in MySQL-like syntax"""
        df = self.spider_foreign[self.spider_foreign['Database name'] == db_name]
        output = "["
        for index, row in df.iterrows():
            output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + \
                      " = " + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ','
        output = output[:-1] + "]" if output != "[" else "[]"
        return output

    def schema_linking(self, question: str, schema: DatabaseSchema, db_id: str) -> str:
        """Step 1: Schema Linking with improved schema validation"""
        print("\nSchema Linking Step:")
        print(f"Question: {question}")
        print(f"Available tables: {list(schema.tables.keys())}")
        # Format schema information more explicitly
        schema_text = "Database Schema:\n"
        for table_name, columns in schema.tables.items():
            schema_text += f"Table {table_name}, columns = [{','.join(columns)}]\n"
        
        print(f'The following are the {schema_text}')

        # Format foreign keys with better readability
        if schema.foreign_keys:
            fk_text = "Foreign Keys:\n"
            for fk in schema.foreign_keys:
                fk_text += f"- {fk[0]} = {fk[1]}\n"
            fk_text += "\nForeign_keys = [" + ",".join(f"{fk[0]} = {fk[1]}" for fk in schema.foreign_keys) + "]"
        else:
            fk_text = "No Foreign Keys\nForeign_keys = []"
        
        print(f'Here are the Foreign Keys {fk_text} \n Those are all the FKs')


        # SQL Schema Linking Task

        prompt = f"""

        You are a SQL database management expert. You are given a database schema (tables, columns, primary keys, foreign keys). You're tasked with finding the schema links to solve a query. Find the ONLY required tables and columns.
    
        Here are examples of Schema Links
        {schema_linking_prompt}

        Here is the Current Schema Information
        {schema_text}
        Here are the foreign keys
        {fk_text}

        The SQL query that needs to be solved: "{question}"

        Instructions:
        1. Analyze the question and identify required tables and columns from the schema above.
        2. ONLY use tables and columns that exist in the schema. DO NOT assume or create tables/columns that aren't listed in the schema.
        3. Include any necessary foreign key relationships.
        4. Do not include unecessary columns and joins for tables.
        4. **IMPORTANT**: Output the schema links in the following format without any Markdown formatting, asterisks, or parentheses: Schema Links: [table1.column1, table2.column2, table1.column3 = table2.column3]

    """
        response = self._get_completion(prompt)
        print(f"LLM Response for schema linking:\n{response}")

        # Extract and validate schema links
        schema_links = self._extract_schema_links(response)
        #validated_links = self._validate_schema_links(schema_links, schema)

        #if validated_links != schema_links:
            #print(f"Warning: Invalid schema links were removed or corrected")
            #print(f"Original: {schema_links}")
            #print(f"Validated: {validated_links}")
            #return validated_links

        return schema_links

    def classification_prompt_maker(self, test_sample_text, database, schema_links):
        instruction = (f'''
        For the given question, classify it as EASY, NON-NESTED, or NESTED based on nested queries and JOIN. Learn from the following examples.
        {classification_prompt}
        \n\n''')

        fields = self.find_fields_MYSQL_like(database)
        fields += "Foreign_keys = " + self.find_foreign_keys_MYSQL_like(database) + '\n'
        fields += "\n"
        prompt = instruction + 'Q: "' + test_sample_text + '\nschema_links: ' + schema_links + """\nA: Let's think step by step."""
        return prompt

    def classify_query(self, question: str, schema_links: str, schema: DatabaseSchema, db_id: str) -> tuple[str, str]:
        """Step 2: Query Classification - returns both classification and full response"""
        classification = None
        while classification is None:
            try:
                prompt = self.classification_prompt_maker(question, db_id, schema_links[1:])
                classification = self._get_completion(prompt)
                print(f'Classification LLM Response ******* {classification} *********')
            except:
                time.sleep(3)
                continue

        try:
            predicted_class = classification.split('Label: "')[1].split('"')[0]
        except:
            print("Slicing error for the classification module")
            predicted_class = "NESTED"

        return predicted_class, classification

    def easy_prompt_maker(self, test_sample_text, database, schema_links):
        instruction = EASY_PROMPT
        # Get the current database schema
        fields = self.find_fields_MYSQL_like(database)
        foreign_keys = "Foreign_keys = " + self.find_foreign_keys_MYSQL_like(database) + '\n'

        prompt_old = f"""{instruction}
User question database schema:
{fields}
{foreign_keys}

# User question
Question: "{test_sample_text}"

# Relevant schema
Schema_links: {schema_links}
"""

        prompt = f"""{instruction}
# User question
Question: "{test_sample_text}"

# Relevant schema
Schema_links: {schema_links}

# Foreign keys
{foreign_keys}

"""
        return prompt, fields, foreign_keys


    def medium_prompt_maker(self, test_sample_text, database, schema_links):
        instruction = MEDIUM_PROMPT

        # Current database schema
        fields = self.find_fields_MYSQL_like(database)
        foreign_keys = "Foreign_keys = " + self.find_foreign_keys_MYSQL_like(database) + '\n'
        prompt_old = f"""{instruction}
User question database schema:
{fields}
{foreign_keys}

# User question
Question: "{test_sample_text}"

# Relevant schema
Schema_links: {schema_links}
"""

        prompt = f"""{instruction}
# User question
Question: "{test_sample_text}"

# Relevant schema
Schema_links: {schema_links}

# Foreign keys
{foreign_keys}

"""
        return prompt, fields, foreign_keys


    def hard_prompt_maker(self, test_sample_text, database, schema_links, sub_questions):
        instruction = HARD_PROMPT

        # Current database schema
        fields = self.find_fields_MYSQL_like(database)
        foreign_keys = "Foreign_keys = " + self.find_foreign_keys_MYSQL_like(database) + '\n'

        stepping = f"""Let's solve "{test_sample_text}" by following the reasoning structure and breaking it down:
        1. Required Schema Links: {schema_links}
        2. Sub-questions to solve: "{sub_questions}"
        3. Using only the tables and columns from our schema links."""

        prompt_old = f"""{instruction}
User question database schema:
{fields}
{foreign_keys}

# User question: 
Question: "{test_sample_text}"

# Relevant schema
Schema_links: {schema_links}

        """

        prompt = f"""{instruction}
# User question: 
Question: "{test_sample_text}"

# Relevant schema
Schema_links: {schema_links}

# Foreign keys
{foreign_keys}

"""
        return prompt, fields, foreign_keys


    def generate_sql(self, question: str, schema_links: str, query_type: str,
                classification_response: str, schema: DatabaseSchema, db_id: str) -> str:

        """Step 3: SQL Generation"""

        # validated_links = self._validate_schema_links(schema_links, schema)

        if query_type == "EASY":
            prompt, fields, foreign_keys = self.easy_prompt_maker(question, db_id, schema_links)
            response = self._get_completion(prompt)
            print(f'SQL Generation Step: The full response is \n {response}')
            # Extract reasoning and SQL
            reasoning, sql = self._extract_reasoning_and_sql(response)
            # Store the reasoning and SQL
            self.store_intermediate_result('sql_reasoning', reasoning)
            self.store_intermediate_result('generated_sql', sql)
            return reasoning, sql, query_type, fields, foreign_keys

        elif query_type == "NON-NESTED":
            prompt, fields, foreign_keys = self.medium_prompt_maker(question, db_id, schema_links)
            response = self._get_completion(prompt)
            print(f'SQL Generation Step: The full response is \n {response}')
            # Extract reasoning and SQL
            reasoning, sql = self._extract_reasoning_and_sql(response)
            # Store the reasoning and SQL
            self.store_intermediate_result('sql_reasoning', reasoning)
            self.store_intermediate_result('generated_sql', sql)
            return reasoning, sql, query_type, fields, foreign_keys

        else:  # NESTED queries
            try:
                sub_questions = classification_response.split('questions = ["')[1].split('"]')[0]
            except:
                print("Error extracting sub-questions, using original question")
                sub_questions = question

            SQL = None # Use 'SQL' in caps for HARD questions
            while SQL is None:
                try:
                    prompt, fields, foreign_keys = self.hard_prompt_maker(question, db_id, schema_links, sub_questions)
                    response = self._get_completion(prompt)
                    print(f'SQL Generation Step: The full response is \n {response}')
                    # Extract reasoning and SQL
                    reasoning, SQL = self._extract_reasoning_and_sql(response)
                    # Store the reasoning and SQL
                    self.store_intermediate_result('sql_reasoning', reasoning)
                    self.store_intermediate_result('generated_sql', SQL)
                    if SQL == "SELECT":  # If extraction failed
                        continue
                except:
                    time.sleep(3)
                    continue
            return reasoning, SQL, query_type, fields, foreign_keys


    def sql_query_corrector(self, test_sample_text, database, reasoning, sql):
        """ Validate and correct SQL query based on schema and question context. """
        instruction = f"""

    #### For the given question, use the provided tables, columns, foreign keys, and primary keys to fix the given SQLite SQL QUERY for any issues. If there are any problems, fix them. If there are no issues, return the SQLite SQL QUERY as is.
    #### Use the following instructions for fixing the SQL QUERY:
    1) Use the database values that are explicitly mentioned in the question.
    2) Pay attention to the columns that are used for the JOIN by using the Foreign_keys.
    3) Use DESC and DISTINCT when needed.
    4) Pay attention to the columns that are used for the GROUP BY statement.
    5) Pay attention to the columns that are used for the SELECT statement.
    6) Only change the GROUP BY clause when necessary (Avoid redundant columns in GROUP BY).
    7) Use GROUP BY on one column only.

    # Ensure that the query is the most succinct version. Explicitly check for redundancy and unecessary JOIN, columns, and aggregations.
    # If there are issues, fix them.
    # If there are no issues, return the SQLite SQL QUERY as is.

    #### Only make changes to the SQL query if there are actual errors. Do not modify the SQL query if it is already correct.

    #### IMPORTANT: 
    {system_prompt_sql}

    """
        fields = self.find_fields_MYSQL_like(database)
        fields += "Foreign_keys = " + self.find_foreign_keys_MYSQL_like(database) + '\n'
        fields += "Primary_keys = " + self.find_primary_keys_MYSQL_like(database)
        prompt = (instruction + '\n' + fields + '\n#### Question: ' + test_sample_text + '\n#### Reasoning steps\n' + reasoning
        + '\n#### SQLite QUERY\n' + sql + '\n#### SQLite FIXED SQL QUERY\n<REASONING>\n')
        print(f'**** The fields for self correction are {fields}')
        
        return prompt

    def self_correction(self, reasoning: str, sql: str, question: str, schema: DatabaseSchema, db_id: str) -> Tuple[str, str]:
        """Step 4: SQL Query Correction"""
        prompt = self.sql_query_corrector(test_sample_text=question, database=db_id, reasoning=reasoning, sql=sql)
        response = self._get_completion(prompt)
        corrected_reasoning, corrected_sql = self._extract_reasoning_and_sql(response)
        
        # Store the reasoning and corrected SQL
        self.store_intermediate_result('corrected_reasoning', corrected_reasoning)
        self.store_intermediate_result('corrected_sql', corrected_sql)
        
        return corrected_reasoning, corrected_sql

    def _extract_reasoning_and_sql(self, response: str) -> Tuple[str, str]:
        """Extract reasoning and SQL from response."""
        pattern = ""
        
        if "<REASONING>" in response:
            pattern = re.compile(
                r'<REASONING>\s*(.*?)\s*</REASONING>.*?<SQL>\s*(.*?)\s*</SQL>',
                re.DOTALL | re.IGNORECASE
            )
        if "<chains>" in response:
            pattern = re.compile(
                r'<chains>\s*(.*?)\s*</chains>.*?<SQL>\s*(.*?)\s*</SQL>',
                re.DOTALL | re.IGNORECASE
            )

        # pattern = re.compile(
        #     r'<REASONING>\s*(.*?)\s*</REASONING>.*?<SQL>\s*(.*?)\s*</SQL>',
        #     re.DOTALL | re.IGNORECASE
        # )
        match = pattern.search(response)
        if match:

            # Strip reasoning and sql responses from tags
            reasoning = match.group(1).strip()
            sql = match.group(2).strip()

            # Remove the excess white space and \n characters
            reasoning = ' '.join(reasoning.split())
            sql = ' '.join(sql.split())

            # Capitalize SQL Characters
            keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'GROUP BY', 'ORDER BY', 'LIMIT', 'HAVING', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
            for kw in keywords:
                sql = re.sub(r'\b' + kw + r'\b', kw, sql, flags=re.IGNORECASE)

            return reasoning, sql
        
        else:
            # If tags are not found, handle accordingly
            print("No <REASONING> and/or <SQL> tags found in the response.")
            return response.strip(), "SELECT"

    def _get_completion(self, prompt: str) -> str:
        """Get completion from OpenAI API with retry logic"""
        max_retries = 3
        delay = 20

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=1000
                )

                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Retrying after error: {str(e)}")
                    time.sleep(delay)
                else:
                    raise e

    def _validate_schema_links(self, schema_links: str, schema: DatabaseSchema) -> str:
        """Validate that schema links only reference existing tables and columns"""
        # Remove brackets and split by comma
        links = schema_links.strip('[]').split(',')
        valid_links = []

        # Get all valid table names and columns
        valid_tables = set(schema.tables.keys())
        valid_columns = {col: table for table, cols in schema.tables.items() for col in cols}

        for link in links:
            link = link.strip()
            if not link:
                continue

            # Check if it's a foreign key relationship
            if '=' in link:
                left, right = link.split('=')
                left = left.strip()
                right = right.strip()

                # Validate both sides of the relationship
                if '.' in left and '.' in right:
                    left_table, left_col = left.split('.')
                    right_table, right_col = right.split('.')

                    if (left_table in valid_tables and
                            right_table in valid_tables and
                            left_col in schema.tables[left_table] and
                            right_col in schema.tables[right_table]):
                        valid_links.append(f"{left} = {right}")

            # Check if it's a column reference
            elif '.' in link:
                table, column = link.split('.')
                if table in valid_tables and column in schema.tables[table]:
                    valid_links.append(link)

        return f"[{', '.join(valid_links)}]"

    def _extract_schema_links(self, response: str) -> str:
        """Extract and validate schema links"""
        
        try:
            # Find the line that starts with 'Schema Links:'
            lines = response.split('\n')
            schema_links_line = ''
            for line in lines:
                if line.strip().startswith('Schema Links:'):
                    schema_links_line = line.strip()
                    break
            if not schema_links_line:
                print("Schema links not found in the response.")
                return "[]"

            # Extract the schema links after 'Schema Links:'
            links = schema_links_line[len('Schema Links:'):].strip()

            # Remove any asterisks, parentheses, or extra characters
            links = links.strip('[]()* ')
            links = links.replace('**', '')  # Remove Markdown bold markers if present
            # links = links.replace('_', '')   # Remove Markdown italic markers if present
            links = links.replace('`', '')   # Remove inline code markers if present

            # Re-add square brackets to standardize the format
            return f"[{links}]"
        except Exception as e:
            print(f"Error extracting schema links: {str(e)}")
            return "[]"

    def process_question(self, question: str, db_id: str, schemas: Dict[str, DatabaseSchema]) -> str:

        """Process a single question through the full pipeline"""

        try:
            schema = schemas[db_id]

            # Step 1: Schema Linking - print only once
            schema_links = self.schema_linking(question, schema, db_id)
            print(f'Schema Links {schema_links}')

            # Step 2: Query Classification - print only once
            query_type, classification_response = self.classify_query(question, schema_links, schema, db_id)
            print(f"Query Type: {query_type}")

            # Step 3: SQL Generation - single detailed print
            reasoning, generated_query, generated_query_type, fields, foreign_keys = self.generate_sql(question, 
                                                                                                        schema_links, 
                                                                                                        query_type, 
                                                                                                        classification_response, 
                                                                                                        schema, db_id)
            print(f"Generated SQL: {generated_query}")

            # Step 4: Self-correction - only print if different from generated SQL
            # final_reasoning, final_sql = self.self_correction(question=question, schema=schema, db_id=db_id, reasoning=reasoning, sql=generated_query)
            final_reasoning = reasoning
            final_sql = generated_query
            
            if final_sql != generated_query:
                print(f"Corrected SQL: {final_sql}")

            return final_reasoning, final_sql, generated_query_type, schema_links, fields, foreign_keys
            # return "", "", "", schema_links, "", ""

        except Exception as e:
            print(f"\nError processing question: {str(e)}")
            raise


def process_question_batch(args):
    """Process a batch of questions in a separate process"""
    questions_batch, schema_file, process_output_file = args

    random.shuffle(questions_batch)

    # Initialize pipeline for this process
    pipeline = DINSQLPipeline()
    # Load schemas
    schemas = pipeline.load_schema(schema_file)

    # Create schema DataFrames for this process
    pipeline.creating_schema(schema_file)

    batch_results = []

    n_processed = 0  # Counter for the number of questions processed

    for i, q in enumerate(questions_batch):
        try:
            print(f"\nProcessing question {i + 1}")

            if q['db_id'] not in schemas:
                print(f"Warning: No schema found for database {q['db_id']}")
                continue

            reasoning, sql, gen_query_type, schema_links, fields, foreign_keys = pipeline.process_question(q['question'], 
                                                                                                            q['db_id'], 
                                                                                                            schemas)

            if not sql or sql == "SELECT":
                print("Warning: Generated SQL appears to be empty or invalid")

            result = {
                'question': q['question'],
                'schema_links': schema_links,
                'fields': fields,
                'foriegn keys': foreign_keys,
                'classification': gen_query_type,
                'predicted_sql': sql,
                'gold_sql': q.get('query', ''),
                'db_id': q['db_id'],
                'reasoning': reasoning
                
            }

            print(f'This reuslt is \n {result}')

            # Append the result to batch_results
            batch_results.append(result)

            # Optionally, write results to file after each question
            with open(process_output_file, 'w') as f:
                json.dump(batch_results, f, indent=2)

            print(f"Processed question {i + 1}")

            # Increment the processed counter
            n_processed += 1

            # Add delay to handle rate limits
            time.sleep(1)

        except Exception as e:
            print(f"\nError processing question {i + 1}: {str(e)}")
            print(f"Question was: {q['question']}")
            print(f"Database was: {q['db_id']}")
            continue

    # Return the number of questions processed for progress bar update
    return n_processed


def process_dataset_parallel(schema_file: str, input_file: str, output_file: str, num_processes: int = 4):
    print("\n=== Starting Parallel Processing ===")

    # Load questions
    print("\n=== Loading Questions ===")
    with open(input_file) as f:
        questions = json.load(f)
    # questions = questions[:1]
    # Calculate batch sizes
    total_questions = len(questions)
    batch_size = total_questions // num_processes or 1
    if batch_size == 0:
        batch_size = 1
        num_processes = total_questions

    # Prepare arguments for each process
    process_args = []
    for i in range(num_processes):
        start_idx = i * batch_size
        # Ensure we cover all questions in the last batch
        end_idx = start_idx + batch_size if i < num_processes - 1 else total_questions
        # Generate a unique output filename for this process
        process_output_file = f"{output_file}_part_{i}.json"
        # Slice the questions list for this batch
        questions_batch = questions[start_idx:end_idx]
        process_args.append((questions_batch, schema_file, process_output_file))

    return process_args, total_questions

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DIN-SQL Pipeline')

    parser.add_argument('--schema', 
                       default="spider/tables.json",
                       help='Path to schema file (default: spider/tables.json)')
    parser.add_argument('--input', 
                       default="spider/dev.json",
                       help='Path to input questions file (default: spider/dev.json)')
    parser.add_argument('--output', 
                       default="din_sql_results.json",
                       help='Path for output results (default: din_sql_results.json)')
    parser.add_argument('--processes',
                       type=int,
                       default=4,
                       help='Number of parallel processes to use (default: 4)')

    args = parser.parse_args()
    print(args)

    # Get process arguments and total questions
    process_args, total_questions = process_dataset_parallel(args.schema, args.input, args.output, args.processes)

    # Create pool and process batches
    print(f"\n=== Processing with {args.processes} processes ===")
    with multiprocessing.Pool(processes=args.processes) as pool:
        with tqdm(total=total_questions, desc="Processing questions") as pbar:
            for n_processed in pool.imap_unordered(process_question_batch, process_args):
                pbar.update(n_processed)

    # Merge the individual output files into the final output file
    all_results = []
    for i in range(args.processes):
        process_output_file = f"{args.output}_part_{i}.json"
        try:
            with open(process_output_file, 'r') as f:
                results = json.load(f)
                all_results.extend(results)
            # Optionally, delete the process's output file
            os.remove(process_output_file)
        except Exception as e:
            print(f"Error reading from {process_output_file}: {str(e)}")

    # Write all results to the final output file
    try:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Successfully wrote results to {args.output}")
    except Exception as e:
        print(f"Error writing to output file: {str(e)}")

    # Print final statistics
    print("\n=== Processing Summary ===")
    print(f"Total questions processed: {total_questions}")
    print(f"Successful generations: {len(all_results)}")
    success_rate = (len(all_results) / total_questions) * 100 if total_questions else 0
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
