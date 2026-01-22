
OPTIONAL MATCH (p1:Person {id:4398046511249})-[edge:guarantee*1..5]->(pN:Person) -[:apply]->(loan:Loan) 
WHERE minInList(getMemberProp(edge, 'timestamp')) > 1643673600000 AND maxInList(getMemberProp(edge, 'timestamp')) < 1646092800000 
WITH DISTINCT loan 
WITH sum(loan.loanAmount) as sumLoanAmount, count(distinct loan) as numLoans 
RETURN round(sumLoanAmount * 1000) / 1000 as sumLoanAmount, numLoans;