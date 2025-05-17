import React from 'react';
import Card from './Card';

function ProfileCard(props) {
    return (
        <Card title="Hwang" backgroundColor="#4ea04e">
            <p>안녕하세요</p>
            <p>리액트를 사용해서 개발하고 있습니다.</p>
        </Card>
    );
}

export default ProfileCard;