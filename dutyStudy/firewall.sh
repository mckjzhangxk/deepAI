iptables -F
iptables -A INPUT -s 140.44.0.0/16 -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -s 140.44.0.0/16 -p tcp --dport 1521 -j ACCEPT
iptables -A INPUT -s 140.44.0.0/16 -p tcp --dport 80 -j ACCEPT

iptables -A INPUT -s 140.16.16.42 -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -s 140.16.16.42 -p tcp --dport 1521 -j ACCEPT
iptables -A INPUT -s 140.16.16.42 -p tcp --dport 80 -j ACCEPT

iptables -A INPUT -s 140.16.16.48 -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -s 140.16.16.48 -p tcp --dport 1521 -j ACCEPT
iptables -A INPUT -s 140.16.16.48 -p tcp --dport 80 -j ACCEPT


iptables -A INPUT -s 140.12.96.53 -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -s 140.12.96.53 -p tcp --dport 1521 -j ACCEPT
iptables -A INPUT -s 140.12.96.53 -p tcp --dport 80 -j ACCEPT

iptables -A INPUT -j DROP

