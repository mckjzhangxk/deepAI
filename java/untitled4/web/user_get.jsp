<%@ page language="java" contentType="text/html; charset=UTF-8"
         pageEncoding="UTF-8"%>
<%@taglib prefix="s" uri="/struts-tags"%>
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    <title>Insert title here</title>
</head>
<body bgcolor="#33CC99">
<h3 align="center">用户列表</h3></body>
<table align="center">
    <tr>
        <td>ID</td>
        <td>用户名</td>
        <td>密码</td>
    </tr>

    <s:iterator value="users" >
        <tr>
            <td> <s:property value="id" />        </td>
            <td> <s:property value="username" />  </td>
            <td> <s:property value="password" />  </td>
        </tr>
    </s:iterator>
</table>

</html>