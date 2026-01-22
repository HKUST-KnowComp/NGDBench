
OPTIONAL MATCH (p1:Person {id:8796093022820})-[edge:guarantee*1..5]->(pN:Person) -[:apply]->(loan:Loan) 
WHERE minInList(getMemberProp(edge, 'timestamp')) > 1646092800000 AND maxInList(getMemberProp(edge, 'timestamp')) < 1650931200000 
WITH DISTINCT loan 
WITH sum(loan.loanAmount) as sumLoanAmount, count(distinct loan) as numLoans 
RETURN round(sumLoanAmount * 1000) / 1000 as sumLoanAmount, numLoans;