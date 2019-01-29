mapping ( address => uint ) userBalances ;
function getBalance ( address u) constant returns ( uint )
return userBalances [u];
function addToBalance ()
userBalances [msg . sender ] += msg . value ;
function withdrawBalance ()
if (!( msg . sender . call . value (userBalances [msg . sender ])()))
userBalances [msg . sender ] = 0;
