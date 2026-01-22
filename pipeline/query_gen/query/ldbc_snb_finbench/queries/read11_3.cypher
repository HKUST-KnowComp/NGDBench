
OPTIONAL MATCH (p1:Person {id:26388279067462})-[edge:guarantee*1..5]->(pN:Person) -[:apply]->(loan:Loan) 
WHERE minInList(getMemberProp(edge, 'timestamp')) > 1625097600000 AND maxInList(getMemberProp(edge, 'timestamp')) < 1627516800000 
WITH DISTINCT loan 
WITH sum(loan.loanAmount) as sumLoanAmount, count(distinct loan) as numLoans 
RETURN round(sumLoanAmount * 1000) / 1000 as sumLoanAmount, numLoans;