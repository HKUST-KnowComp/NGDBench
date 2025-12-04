
OPTIONAL MATCH (p1:Person {id:%d})-[edge:guarantee*1..5]->(pN:Person) -[:apply]->(loan:Loan) 
WHERE minInList(getMemberProp(edge, 'timestamp')) > %d AND maxInList(getMemberProp(edge, 'timestamp')) < %d 
WITH DISTINCT loan 
WITH sum(loan.loanAmount) as sumLoanAmount, count(distinct loan) as numLoans 
RETURN round(sumLoanAmount * 1000) / 1000 as sumLoanAmount, numLoans;